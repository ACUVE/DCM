#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "draw_thread.hpp"
#include "interval_timer.hpp"

bool draw_thread::wait_update( void )
{
	while( last_tracking_index == tdata.tracking_frame.load() )
	{
		if( exit_flag.load() ) return false;
		// std::this_thread::yield();
	}
	return true;
}

void draw_thread::thread_func( void )
{
	if( !wait_update() ) return;

	interval_timer calc_fps( "draw_thread" );

	auto const POINT_NUM = std::accumulate( tdata.point_size.begin(), tdata.point_size.end(), 0u );

	std::vector< cv::Point3f > objpoint;
	std::vector< cv::Point2f > imagepoint;
	objpoint.reserve( POINT_NUM );
	imagepoint.reserve( POINT_NUM );

	auto tmpimg = image_buffer.get();
	cv::Mat rvec, tvec;
	unsigned int last_used_point_num = 0u;

	auto const cluster_data_size = tdata.point_size.size();
	assert( cluster_data_size == 1u );

	while( !exit_flag.load() )
	{
		objpoint.clear();
		imagepoint.clear();

		if( !wait_update() ) break;
		last_tracking_index = tdata.tracking_frame.load();

		for( auto i = 0u; i < cluster_data_size; ++i )
		{
			auto const &paper = tdata.point[ i ];
			auto const paper_cluster_size = tdata.point_size[ i ];
			for( auto j = 0u; j < paper_cluster_size; ++j )
			{
				auto const &point = paper[ j ];
				auto const state = point.state.load( std::memory_order::memory_order_acquire );
				if( state != tracking_state::TRACKING ) continue;
				auto const x = point.x.load( std::memory_order::memory_order_relaxed );
				auto const y = point.y.load( std::memory_order::memory_order_relaxed );
				imagepoint.emplace_back( x, y );
				auto pp = &cluster_point[ i ][ j * 3 ];
				objpoint.emplace_back( pp[ 0 ], pp[ 1 ], pp[ 2 ] );
			}
		}
		auto crrimg = image_buffer.get();

		cv::Mat imgroibin, fuck;
		cv::adaptiveThreshold( crrimg, imgroibin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 9, 8 );
		cv::erode( imgroibin, fuck, cv::noArray() );
		// cv::dilate( fuck, fuck, cv::noArray() );
		cv::imshow( "fuck", fuck );

		bool const useGuess = last_used_point_num >= 6;
		auto const objpoint_size = objpoint.size();
		last_used_point_num = static_cast< unsigned int >( objpoint_size );
		// if( objpoint_size ) std::cout << objpoint_size << std::endl;
		if( objpoint_size >= 3 ) 
		{
			// cv::solvePnP( objpoint, imagepoint, cameraMatrix, cv::noArray(), rvec, tvec, useGuess );
			cv::solvePnPRansac( objpoint, imagepoint, cameraMatrix, cv::noArray(), rvec, tvec, useGuess, 100, 0.01f );
	
			std::vector< cv::Point2f > point;
			cv::projectPoints( objpoint, rvec, tvec, cameraMatrix, cv::noArray(), point );
	
			cv::cvtColor( crrimg, crrimg, cv::COLOR_GRAY2BGR );
			for( auto const &v : point )
			{
				
				cv::circle( crrimg, v, 5, cv::Scalar( 255, 255, 0 ), -1 );
			}
		}
		cv::imshow( "draw", crrimg );
		cv::waitKey( 1 );

		/*
		auto crrimg = image_buffer.get();
		cv::cvtColor( crrimg, crrimg, cv::COLOR_GRAY2BGR );
		
		for( auto i = 0u; i < cluster_data_size; ++i )
		{
			auto const &paper = tdata.point[ i ];
			auto const paper_cluster_num = tdata.point_size[ i ];
			for( auto j = 0u; j < paper_cluster_num; ++j )
			{
				auto const &point = paper[ j ];
				auto const state = point.state.load( std::memory_order::memory_order_acquire );
				if( state != tracking_state::TRACKING ) continue;
				// if( state != tracking_state::TRACKING && state != tracking_state::BEFORE_TRACKING ) continue;
				auto x = point.x.load( std::memory_order::memory_order_relaxed );
				auto y = point.y.load( std::memory_order::memory_order_relaxed );
				auto age = point.age.load( std::memory_order::memory_order_relaxed );
				auto ix = static_cast< int >( x ), iy = static_cast< int >( y );
				cv::Scalar color;
				if( age == 0 ) color = cv::Scalar( 0, 0, 255 );
				else color = cv::Scalar( 255 / age, 255 / age, 0 );
				cv::circle( crrimg, cv::Point( ix, iy ), 5, color, -1 );
			}
		}
		
		// auto tmpimg = cv::Mat::zeros( cv::Size( crrimg.cols, crrimg.rows ), CV_8UC3 );
		cv::imshow( "draw", crrimg );
		cv::waitKey( 1 );
		// */

		calc_fps.interval();
	}
}
