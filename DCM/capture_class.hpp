#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thread>

class video_capture
{
public:
	video_capture()
		: cap( 0 )
	{
		cap >> tmp;
	}
	void capture( cv::Mat &img )
	{
		cap >> tmp;
		cv::cvtColor( tmp, img, cv::COLOR_RGB2GRAY );
	}

	unsigned int width() const
	{
		return tmp.cols;
	}
	unsigned int height() const
	{
		return tmp.rows;
	}

private:
	cv::VideoCapture cap;
	cv::Mat tmp;
};

class cv_mat_capture
{
public:
	// vec.empty() must be false
	cv_mat_capture( std::vector< cv::Mat > vec )
		: index( 0u )
		, vec( std::move( vec ) )
		, w( this->vec.front().cols )
		, h( this->vec.back().rows )
	{
#if _DEBUG
		for( auto &&m : vec )
		{
			if( w != m.cols || h != m.rows)
			{
				throw std::exception( "error" );
			}
		}
#endif
	}
	void capture( cv::Mat &img )
	{
		img = vec[ index++ % vec.size() ].clone();
	}

	unsigned int width() const
	{
		return w;
	}
	unsigned int height() const
	{
		return h;
	}

private:
	std::size_t index;
	std::vector< cv::Mat > vec;
	unsigned int const w;
	unsigned int const h;
};

// #if __has_include( <xiApi.h> )
#include <windows.h>
#include <xiApi.h>
#pragma comment( lib, "m3apiX64.lib" )

class ximea_xiq_capture
{
public:
	ximea_xiq_capture( float const fps = 100.0f )
	{
		#define XI( func, args ) if( func args != XI_OK ) throw std::runtime_error( #func #args " error" );

		DWORD camera_num;
		XI( xiGetNumberDevices, ( &camera_num ) );
		if( camera_num == 0u )
		{
			std::cerr << "No camera." << std::endl;
			throw std::runtime_error( "No camera." );
		}

		XI( xiOpenDevice, ( 0, &xi_handle ) );
		// “K“–fps
		XI( xiSetParamInt, ( xi_handle, XI_PRM_TRG_SOURCE, XI_TRG_OFF ) );
		XI( xiSetParamInt, ( xi_handle, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FRAME_RATE ) );
		// ƒgƒŠƒK
		// XI( xiSetParamInt, ( xi_handle, XI_PRM_TRG_SOURCE, XI_TRG_EDGE_RISING ) );
		XI( xiSetParamInt, ( xi_handle, XI_PRM_EXPOSURE, static_cast< int >( 1000000.0f / fps - 100 ) ) );
		XI( xiSetParamFloat, ( xi_handle, XI_PRM_FRAMERATE, fps ) );
		XI( xiStartAcquisition, ( xi_handle ) );
		
		XI( xiGetImage, ( xi_handle, 50000, &xi_img ) );
		w = xi_img.width, h = xi_img.height;
		#undef XI
	}
	ximea_xiq_capture( ximea_xiq_capture const & ) = delete;
	ximea_xiq_capture( ximea_xiq_capture &&r )
		: xi_handle( r.xi_handle )
		, w( r.w ), h( r.h )
		, xi_img( r.xi_img )
	{
		r.xi_handle = nullptr;
	}
	~ximea_xiq_capture()
	{
		if( xi_handle != nullptr )
		{
			xiStopAcquisition( xi_handle );
			xiCloseDevice( xi_handle );
		}
	}
	ximea_xiq_capture &operator=( ximea_xiq_capture & ) = delete;
	ximea_xiq_capture &operator=( ximea_xiq_capture && r )
	{
		xi_handle = r.xi_handle;
		w = r.w; h = r.h;
		xi_img = r.xi_img;
		r.xi_handle = nullptr;
	}
	void capture( cv::Mat &img )
	{
		img.create( h, w, CV_8UC1 );
		xiGetImage( xi_handle, 50000, &xi_img );
		std::memcpy( img.data, xi_img.bp, w * h );
	}
	unsigned int width() const
	{
		return w;
	}
	unsigned int height() const
	{
		return h;
	}

private:
	HANDLE xi_handle = nullptr;
	unsigned int w = 0u, h = 0u;
	XI_IMG xi_img = { sizeof( xi_img ) };
};
// #endif
