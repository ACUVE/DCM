#pragma once

#include <thread>
#include <opencv2/opencv.hpp>
#include "tracking_data.hpp"
#include "ring_buffer.hpp"

class tracking_thread
{
private:
	tracking_data &tdata;
	ring_buffer< cv::Mat > const &image_buffer;
	std::size_t last_image_index;
	std::thread thread;
	std::atomic< bool > exit_flag;
	// std::get< 0 >( cluster_data ) ‚Í index ‚Ì”z—ñ, std::get< 1 >( cluster_data ) ‚Í num ‚Ì”z—ñ
	std::vector< std::tuple< std::vector< unsigned int >, std::vector< unsigned int > > > const cluster_data;

public:
	tracking_thread( tracking_data &_tdata, ring_buffer< cv::Mat > const &_image_buffer, std::vector< std::tuple< std::vector< unsigned int >, std::vector< unsigned int > > > _cluster_data )
		: tdata( _tdata )
		, image_buffer( _image_buffer )
		, last_image_index( _image_buffer.current_index() )
		, exit_flag( false )
		, cluster_data( std::move( _cluster_data) )
	{
		// do nothing
	}
	void run( void )
	{
		exit_flag = false;
		thread = std::thread( &tracking_thread::thread_func, this );
	}
	void stop( void )
	{
		exit_flag = true;
		thread.join();
	}

private:
	std::size_t wait_next_image( void );
	void thread_func( void );
};
