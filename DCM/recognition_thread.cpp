#include <limits>
#include <chrono>
#include "recognition_thread.hpp"
#include "recognition.hpp"
#include "tracking_data.hpp"
#include "interval_timer.hpp"

#define RCOGNITION_THREAD_DEBUG_SHOW 1

using namespace recognition;

constexpr float AREA_THRESHOLD_MIN = 2.0f;
constexpr float AREA_THRESHOLD_MAX = 40.0f;
constexpr std::size_t KEYPOINT_GRID_NUM_THRESHOLD = 15;
constexpr float KEYPOINT_DISTANCE_THRESHOLD = 15.0f;
constexpr unsigned int MAX_NUM_OF_KEY_POINT_IN_CLUSTER = 6; // 本来は定数で与えられる値ではない．

constexpr std::size_t ABORT_WAIT_INDEX = std::numeric_limits< std::size_t >::max();

std::size_t recognition_thread::wait_next_image( void )
{
	while( !exit_flag.load() )
	{
		std::size_t const ici = image_buffer.current_index();
		if( ici != last_image_index )
		{
			return ici;
		}
		// std::this_thread::yield();
	}
	return ABORT_WAIT_INDEX;
}

void recognition_thread::thread_func( void )
{
	std::vector< key_point > kps;
	std::vector< cluster > clus;
	std::vector< GEOM_FADE2D::Triangle2 * > triangle;
	std::unique_ptr< GEOM_FADE2D::Fade_2D > fade2d;
	std::unordered_map< Cluster_Num_Tuple, Cluster_Index_Tuple, hash_cnt > num_to_index;
	std::vector< Cluster_Index > cluster_index;

	interval_timer calc_fps( "recognition_thread" );

	get_num_to_cluster_id( cluster_data, num_to_index );

	while( !exit_flag.load() )
	{
		auto const image_index = wait_next_image();
		if( image_index == ABORT_WAIT_INDEX ) break;
		last_image_index = image_index;
		auto img_gray = image_buffer.get_with_index( image_index );
		// std::cout << "recognition_thread: image_index = " << image_index << std::endl;

		// デバッグ用画像
#if RCOGNITION_THREAD_DEBUG_SHOW
		cv::Mat key_point_img, cluster_img, triangle_img;
		cv::cvtColor( img_gray, key_point_img, cv::COLOR_GRAY2BGR );
		cluster_img = key_point_img.clone();
		triangle_img = key_point_img.clone();
#endif

		get_key_point( kps, img_gray, AREA_THRESHOLD_MIN, AREA_THRESHOLD_MAX );

#if RCOGNITION_THREAD_DEBUG_SHOW
		draw_key_point( key_point_img, kps );
		cv::imshow( "key_point", key_point_img );
#endif

		get_cluster( clus, kps, KEYPOINT_DISTANCE_THRESHOLD, KEYPOINT_GRID_NUM_THRESHOLD, MAX_NUM_OF_KEY_POINT_IN_CLUSTER, img_gray.cols, img_gray.rows );

#if RCOGNITION_THREAD_DEBUG_SHOW
		draw_cluster( cluster_img, clus );
		cv::imshow( "cluster", cluster_img );
#endif
		get_triangulation( triangle, fade2d, clus );
		/*
		auto rit = std::remove_if(
			triangle.begin(), triangle.end(),
			[]( auto const &tri )
			{
				for( auto i = 0; i < 3; ++i )
				{
					auto sqlen = tri->getSquaredEdgeLength( i );
					if( sqlen >= 7000 ) return true;
				}
				return false;
			}
		);
		triangle.erase( rit, triangle.end() );
		*/

#if RCOGNITION_THREAD_DEBUG_SHOW
		draw_triangle( triangle_img, triangle );
		cv::imshow( "triangle", triangle_img );
#endif

		get_cluster_index( cluster_index, num_to_index, clus, triangle );

		// 取り敢えずここらへんに書いておくけど，recognition.cppに写すべきかもしれない
		for( auto i = 0u; i < std::size( cluster_index ); ++i )
		{
			auto const &pno = std::get< 0 >( cluster_index[ i ] );
			auto const &ind = std::get< 1 >( cluster_index[ i ] );
			if( pno == recognition::INVALID_INDEX ) continue;
			auto const &clusi = clus[ i ];
			auto &p = tdata.point[ pno ][ ind ];
			auto const state = p.state.load( std::memory_order::memory_order_acquire );
			if( state == tracking_state::NO_TRACKING )
			{
				p.x.store( clusi.x, std::memory_order::memory_order_relaxed );
				p.y.store( clusi.y, std::memory_order::memory_order_relaxed );
				p.sx.store( clusi.sx, std::memory_order::memory_order_relaxed );
				p.sy.store( clusi.sy, std::memory_order::memory_order_relaxed );
				p.age.store( 0u, std::memory_order::memory_order_relaxed );
				p.state.store( tracking_state::BEFORE_TRACKING, std::memory_order::memory_order_release );
			}
			else
			{
				auto const x = p.x.load( std::memory_order::memory_order_relaxed );
				auto const y = p.y.load( std::memory_order::memory_order_relaxed );
				auto const dx = x - clusi.x, dy = y - clusi.y;
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
		tdata.recognition_frame.store( last_image_index, std::memory_order::memory_order_seq_cst );

#if RCOGNITION_THREAD_DEBUG_SHOW
		for( int i = 0; i < 100; ++i )
		{
			if( exit_flag.load() ) break;
			cv::waitKey( 10 );
		}
#endif
		calc_fps.interval();
	}
}
