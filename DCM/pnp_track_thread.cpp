#include "pnp_track_thread.hpp"
#include "findMarker.hpp"
#include "make_point.hpp"
#include "interval_timer.hpp"

#define PNP_TRACK_THREAD_DEBUG_SHOW 1

bool pnp_track_thread::wait_tracking_update( void )
{
	while( !exit_flag.load() )
	{
		if( tdata.tracking_frame.load() != last_image_index)
		{
			return true;
		}
		// std::this_thread::yield();
	}
	return false;
}
void pnp_track_thread::thread_func( void )
{
	interval_timer calc_fps( "pnp_track_thread" );

	std::vector< cv::Point3f > objpoint;
	std::vector< cv::Point2f > imagepoint;
	cv::Mat rvec, tvec;

	auto const cluster_data_size = std::size( tdata.point_size );
	assert( cluster_data_size == 1u );
	auto const point_num = tdata.point_size[ 0 ];
	std::vector< cv::Point3f > allPoint;
	std::vector< cv::Point2f > projectedPoint;
	for( auto i = 0u; i < point_num; ++i )
	{
		auto const * const pp = &cluster_point[ 0 ][ i * 3 ];
		allPoint.emplace_back( pp[ 0 ], pp[ 1 ], pp[ 2 ] );
	}
	auto const dual = make_dual( cluster_point[ 0 ], std::get< 0 >( cluster_data[ 0 ] ) );
	auto const &dual_point = std::get< 0 >( dual );
	auto const &dual_index = std::get< 1 >( dual );
	assert( dual_index.size() == point_num );
	std::vector< Vector > normal_vector;
	normal_vector.reserve( point_num );
	for( auto i = 0u; i < point_num; ++i )
	{
		normal_vector.emplace_back( calc_normal_vector( dual_point, dual_index[ i ] ) );
	}

	while( !exit_flag.load() )
	{
		objpoint.clear();
		imagepoint.clear();

		if( !wait_tracking_update() ) break;
		last_image_index = tdata.tracking_frame.load();
		auto crrimg = image_buffer.get_with_index( last_image_index );
		
		auto max_sx = -std::numeric_limits< float >::max(), max_sy = max_sx;
		for( auto j = 0u; j < point_num; ++j )
		{
			auto const &point = tdata.point[ 0 ][ j ];
			auto const state = point.state.load( std::memory_order::memory_order_acquire );
			if( state != tracking_state::TRACKING ) continue;
			if( std::get< 1 >( cluster_data[ 0 ] )[ j ] == 1u ) continue;
			auto const x = point.x.load( std::memory_order::memory_order_relaxed );
			auto const y = point.y.load( std::memory_order::memory_order_relaxed );
			max_sx = std::max( max_sx, point.sx.load( std::memory_order::memory_order_relaxed ) );
			max_sy = std::max( max_sy, point.sy.load( std::memory_order::memory_order_relaxed ) );
			imagepoint.emplace_back( x, y );
			auto pp = &cluster_point[ 0 ][ j * 3 ];
			objpoint.emplace_back( pp[ 0 ], pp[ 1 ], pp[ 2 ] );
		}

#if PNP_TRACK_THREAD_DEBUG_SHOW
			cv::Mat rgb;
			cv::cvtColor( crrimg, rgb, cv::COLOR_GRAY2BGR );
#endif
		auto const objpoint_size = std::size( objpoint );
		if( objpoint_size >= 3 )
		{
			cv::solvePnP( objpoint, imagepoint, cameraMatrix, cv::noArray(), rvec, tvec );
			cv::projectPoints( allPoint, rvec, tvec, cameraMatrix, cv::noArray(), projectedPoint );

			auto const max_sxy = std::max( max_sx, max_sy );

			cv::Mat dst;
			cv::Rodrigues( rvec, dst );
			for( auto i = 0u; i < point_num; ++i )
			{
				auto &p = tdata.point[ 0 ][ i ];
				auto const &v = projectedPoint[ i ];
				auto const &nv = normal_vector[ i ];
				auto const nx = static_cast< float >( dst.at< double >( 0, 0 ) * nv.x + dst.at< double >( 0, 1 ) * nv.y + dst.at< double >( 0, 2 ) * nv.z );
				auto const ny = static_cast< float >( dst.at< double >( 1, 0 ) * nv.x + dst.at< double >( 1, 1 ) * nv.y + dst.at< double >( 1, 2 ) * nv.z );
				auto const nz = static_cast< float >( dst.at< double >( 2, 0 ) * nv.x + dst.at< double >( 2, 1 ) * nv.y + dst.at< double >( 2, 2 ) * nv.z );
 				// std::cout << nx << ", " << ny << ", " << nz << std::endl;
				if( nz > -0.2 ) continue;
#if PNP_TRACK_THREAD_DEBUG_SHOW
				cv::circle( rgb, v, 5, cv::Scalar( 255, 255, 0 ), -1 );
#endif
				float x, y, sx, sy;
				unsigned kpn;
				// if( !findMarker( crrimg, v.x, v.y, max_sxy * (1.0f - std::abs( nx )) * 8.5f, max_sxy * (1.0f - std::abs( ny )), x, y, sx, sy, kpn, "pnp" ) ) continue;
				if( !findMarker( crrimg, v.x, v.y, 60.0f, 60.0f, x, y, sx, sy, kpn, "pnp" ) ) continue;
				if( kpn != std::get< 1 >( cluster_data[ 0 ] )[ i ] ) continue;
				auto const state = p.state.load( std::memory_order::memory_order_acquire );
				if( state == tracking_state::NO_TRACKING )
				{
					p.x.store( x, std::memory_order::memory_order_relaxed );
					p.y.store( y, std::memory_order::memory_order_relaxed );
					p.sx.store( sx, std::memory_order::memory_order_relaxed );
					p.sy.store( sy, std::memory_order::memory_order_relaxed );
					p.age.store( 0u, std::memory_order::memory_order_relaxed );
					p.state.store( tracking_state::BEFORE_TRACKING, std::memory_order_release );
				}
				else
				{
					auto const cx = p.x.load( std::memory_order::memory_order_relaxed );
					auto const cy = p.y.load( std::memory_order::memory_order_relaxed );
					auto const dx = cx - x, dy = cy - y;
					if( dx * dx + dy * dy >= 20 * 20 )
					{
						p.state.store( tracking_state::NO_TRACKING, std::memory_order::memory_order_seq_cst );
						p.x.store( 0.0f, std::memory_order::memory_order_relaxed );
						p.y.store( 0.0f, std::memory_order::memory_order_relaxed );
						p.sx.store( 0.0f, std::memory_order::memory_order_relaxed );
						p.sy.store( 0.0f, std::memory_order::memory_order_relaxed );
						p.age.store( 0u, std::memory_order::memory_order_relaxed );
					}
				}
			}
#if PNP_TRACK_THREAD_DEBUG_SHOW
			for( auto const &v : imagepoint )
			{
				cv::circle( rgb, v, 5, cv::Scalar( 255, 0, 255 ), -1 );
			}
#endif
		}
#if PNP_TRACK_THREAD_DEBUG_SHOW
		cv::imshow( "pnp", rgb );
		cv::waitKey( 1 );
#endif
		calc_fps.interval();
	}
}