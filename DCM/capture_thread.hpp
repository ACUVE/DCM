#pragma once

#include <utility>
#include <atomic>
#include <thread>
#include <tuple>
#include <opencv2/core.hpp>
#include "ring_buffer.hpp"

template< typename Cap >
class capture_thread
{
private:	
	ring_buffer< cv::Mat > &buffer;
	Cap cap;
	std::thread thread;
	std::atomic< bool > exit_flag;
	unsigned int const sleep_time_ms;

public:
	capture_thread( ring_buffer< cv::Mat > &buffer, unsigned int _sleep_time_ms = 0u )
		: buffer( buffer )
		, cap()
		, exit_flag( false )
		, sleep_time_ms( _sleep_time_ms )
	{
	}
	capture_thread( ring_buffer< cv::Mat > &buffer, Cap cap, unsigned int _sleep_time_ms = 0u )
		: buffer( buffer )
		, cap( std::move( cap ) )
		, exit_flag( false )
		, sleep_time_ms( _sleep_time_ms )
	{
	}
	void run( void )
	{
		exit_flag = false;
		thread = std::thread( &capture_thread::thread_func, this );
	}
	void stop( void )
	{
		exit_flag = true;
		thread.join();
	}

	unsigned int width() const
	{
		return cap.width();
	}
	unsigned int height() const
	{
		return cap.height();
	}

private:
	void thread_func( void )
	{
		while( !exit_flag.load() )
		{
			auto &img = buffer.get_next_index();
			cap.capture( img );
			buffer.next_index();
			if( sleep_time_ms )
			{
				std::this_thread::sleep_for( std::chrono::milliseconds( sleep_time_ms ) );
			}
		}
	}
};