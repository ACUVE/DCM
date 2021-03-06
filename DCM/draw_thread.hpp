#pragma once

#include <atomic>
#include <thread>
#include <opencv2/core.hpp>
#include "tracking_data.hpp"
#include "ring_buffer.hpp"

class draw_thread
{
private:
	tracking_data &tdata;
	ring_buffer< cv::Mat > const &image_buffer;
	std::thread thread;
	std::atomic< bool > exit_flag;
	std::size_t last_tracking_index;
	// std::get< 0 >( cluster_data ) は index の配列, std::get< 1 >( cluster_data ) は num の配列
	std::vector< std::tuple< std::vector< unsigned int >, std::vector< unsigned int > > > const cluster_data;
	// 上にまとめるべきなんだろうけど，他のクラスとの兼ね合い上取り敢えずこんな感じ
	std::vector< std::vector< float > > const cluster_point;
	cv::Mat cameraMatrix;

public:
	draw_thread( tracking_data &_tdata, ring_buffer< cv::Mat > const &_image_buffer, std::vector< std::tuple< std::vector< unsigned int >, std::vector< unsigned int > > > _cluster_data, std::vector< std::vector< float > > _cluster_point, cv::Mat _cameraMatrix )
		: tdata( _tdata )
		, image_buffer( _image_buffer )
		, exit_flag( false )
		, last_tracking_index( 0u )
		, cluster_data( std::move( _cluster_data ) )
		, cluster_point( std::move( _cluster_point ) )
		, cameraMatrix( std::move( _cameraMatrix ) )
	{
		assert( cluster_data.size() == cluster_point.size() );
	}

	void run( void )
	{
		exit_flag = false;
		thread = std::thread( &draw_thread::thread_func, this );
	}
	void stop( void )
	{
		exit_flag = true;
		thread.join();
	}

private:
	bool wait_update( void );
	void thread_func( void );
};