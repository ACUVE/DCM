#include <limits>
#include "tracking_thread.hpp"
#include "tracking_data.hpp"
#include "interval_timer.hpp"
#include "findMarker.hpp"

#define TRACKING_THREAD_DEBUG_SHOW 0
#define TRACKING_NO_WAIT_MODE 0

constexpr std::size_t ABORT_WAIT_INDEX = std::numeric_limits< std::size_t >::max();

std::size_t tracking_thread::wait_next_image( void )
{
	while( !exit_flag.load() )
	{
#if !TRACKING_NO_WAIT_MODE
		std::size_t const ici = image_buffer.current_index();
		if( ici != last_image_index )
		{
			return ici;
		}
		// std::this_thread::yield();
#else
		return image_buffer.current_index();
#endif
	}
	return ABORT_WAIT_INDEX;
}

void tracking_thread::thread_func( void )
{
	interval_timer calc_fps( "tracking_thread" );

	while( image_buffer.current_index() < 1 && !exit_flag.load() );
	auto tmp = image_buffer.get();

	// 重複検出用データ
	using CHECK_UINT = std::uint16_t;
	constexpr std::uint8_t CHECK_SHIFT = 2;
	constexpr std::uint8_t CHECK_UINT_BIT = CHAR_BIT * sizeof( CHECK_UINT );
	constexpr std::uint8_t CHECK_PAPER_BIT = CHECK_UINT_BIT / 2, CHECK_POINT_BIT = CHECK_UINT_BIT - CHECK_PAPER_BIT;
	constexpr CHECK_UINT CHECK_POINT_MASK = (1u << CHECK_POINT_BIT) - 1u;
	constexpr CHECK_UINT CHECK_INVALID_NUM = std::numeric_limits< CHECK_UINT >::max();
	auto const CHECK_MAX_X_PLUS_ONE = (tmp.cols >> CHECK_SHIFT) + 1, CHECK_MAX_Y_PLUS_ONE = (tmp.rows >> CHECK_SHIFT) + 1;
	auto const CHECK_DOUBLE_CHECK_SIZE = CHECK_MAX_X_PLUS_ONE * CHECK_MAX_Y_PLUS_ONE;
	std::vector< CHECK_UINT > double_check( CHECK_DOUBLE_CHECK_SIZE, CHECK_INVALID_NUM );
	assert( tdata.point.size() < (1u << CHECK_PAPER_BIT) && std::all_of( tdata.point_size.begin(), tdata.point_size.end(), [ & ]( auto v ){ return v < (1u << CHECK_POINT_BIT); } ) );
	// 重複検出用データここまで

	// 速度
	constexpr float VELO_COEFF = 0.4f;
	std::vector< std::unique_ptr< std::tuple< float, float >[] > > velocity;
	for( auto const size : tdata.point_size ) velocity.emplace_back( std::make_unique< std::tuple< float, float >[] >( size ) );

	while( !exit_flag.load() )
	{
		std::fill( double_check.begin(), double_check.end(), CHECK_INVALID_NUM );
		auto const image_index = wait_next_image();
		if( image_index == ABORT_WAIT_INDEX ) break;
		last_image_index = image_index;
		
		auto img_gray = image_buffer.get_with_index( image_index );
		
		// あるのトラッキング
		auto const tdata_point_size = std::size( tdata.point );
		for( auto i = 0u; i < tdata_point_size; ++i )
		{
			auto &paper = tdata.point[ i ];
			auto const &num_paper = std::get< 1 >( cluster_data[ i ] );
			auto const paper_size = tdata.point_size[ i ];
			for( auto j = 0u; j < paper_size; ++j )
			{
				auto &point = paper[ j ];
				auto &velo = velocity[ i ][ j ];
				auto const tstate = point.state.load( std::memory_order::memory_order_acquire );
				if( !(tstate == tracking_state::BEFORE_TRACKING || tstate == tracking_state::TRACKING) ) continue;
				if( tstate == tracking_state::BEFORE_TRACKING ) velo = std::make_tuple( 0.0f, 0.0f );
				unsigned int found_num;
				float const crr_x = point.x.load( std::memory_order::memory_order_relaxed), crr_y = point.y.load( std::memory_order::memory_order_relaxed );
				float new_x, new_y, new_sx, new_sy;
				auto const sx = point.sx.load( std::memory_order::memory_order_relaxed );
				auto const sy = point.sy.load( std::memory_order::memory_order_relaxed );
				if( !findMarker( img_gray, crr_x + std::get< 0 >( velo ), crr_y + std::get< 1 >( velo ), std::max( sx * 8.5f, 30.0f ), std::max( sy * 8.5f, 30.0f ), new_x, new_y, new_sx, new_sy, found_num, "tracking" ) ) goto track_fail;
				if( num_paper[ j ] != found_num ){
					// std::cout << "paper = " << num_paper[ j ] << ", found = " << found_num << std::endl;
					goto track_fail;
				}
				// gotoのために一段インデント
				{
					// 重複チェック
					auto const ix = static_cast< unsigned int >( new_x ) >> CHECK_SHIFT, iy = static_cast< unsigned int >( new_y ) >> CHECK_SHIFT;
					auto &double_check_crr = double_check[ ix + iy * CHECK_MAX_X_PLUS_ONE ];
					if( double_check_crr != CHECK_INVALID_NUM )
					{
						// 重複していた
						auto &dupp = tdata.point[ double_check_crr >> CHECK_POINT_BIT ][ double_check_crr & CHECK_POINT_MASK ];
						dupp.state.store( tracking_state::NO_TRACKING, std::memory_order::memory_order_seq_cst );
						dupp.x.store( 0.0f, std::memory_order::memory_order_relaxed );
						dupp.y.store( 0.0f, std::memory_order::memory_order_relaxed );
						dupp.sx.store( 0.0f, std::memory_order::memory_order_relaxed );
						dupp.sy.store( 0.0f, std::memory_order::memory_order_relaxed );
						dupp.age.store( 0u, std::memory_order::memory_order_relaxed );
						goto track_fail;
					}
					double_check_crr = (i << CHECK_POINT_BIT) | j;
					// 重複チェックここまで
					point.x.store( new_x, std::memory_order::memory_order_relaxed );
					point.y.store( new_y, std::memory_order::memory_order_relaxed );
					point.sx.store( new_sx, std::memory_order::memory_order_relaxed );
					point.sy.store( new_sy, std::memory_order::memory_order_relaxed );
					point.age.fetch_add( 1u, std::memory_order::memory_order_relaxed );
					point.state.store( tracking_state::TRACKING, std::memory_order_release );
					velo = std::make_tuple(
						(std::get< 0 >( velo ) * VELO_COEFF + (new_x - crr_x)) / (1.0f + VELO_COEFF),
						(std::get< 1 >( velo ) * VELO_COEFF + (new_y - crr_y)) / (1.0f + VELO_COEFF)
					);
				}
				continue;
			track_fail:
				point.state.store( tracking_state::NO_TRACKING, std::memory_order::memory_order_seq_cst );
				point.x.store( 0.0f, std::memory_order::memory_order_relaxed );
				point.y.store( 0.0f, std::memory_order::memory_order_relaxed );
				point.sx.store( 0.0f, std::memory_order::memory_order_relaxed );
				point.sy.store( 0.0f, std::memory_order::memory_order_relaxed );
				point.age.store( 0u, std::memory_order::memory_order_relaxed );
			}
		}
		tdata.tracking_frame.store( last_image_index, std::memory_order::memory_order_seq_cst );

#if TRACKING_THREAD_DEBUG_SHOW
		cv::waitKey( 1 );
#endif

		calc_fps.interval();
	}
}
