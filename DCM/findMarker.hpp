#pragma once

#include <opencv2/opencv.hpp>
#include <limits>
#include <thread>
#include <string>

#define FIND_MARKDER_DEBUG_SHOW 0

namespace
{
	inline
	bool key_point_from_contour( float &x, float &y, float &sx, float &sy, std::vector< cv::Point > const &contour, float const area_threshold_min = -std::numeric_limits< float >::max(), float const area_threshold_max = std::numeric_limits< float >::max() )
	{
		// ‚±‚±’x‚»‚¤
		// https://en.wikipedia.org/wiki/Image_moment#Central_moments
		auto const m = cv::moments( contour, true );
		auto const area = static_cast< float >( m.m00 );
		if( area < area_threshold_min || area_threshold_max < area )
		{
			std::cout << "area drop: " << area << std::endl;
			return false;
		};
		auto const x_bar = m.m10 / m.m00, y_bar = m.m01 / m.m00;
		auto const mu00 = m.m00;
		auto const mu20 = m.m20 - x_bar * m.m10;
		auto const mu02 = m.m02 - y_bar * m.m01;
		auto const mu11_dash = m.m11 / m.m00 - x_bar * y_bar;
		auto const mu20_dash = mu20 / mu00;
		auto const mu02_dash = mu02 / mu00;
		auto const mu20_dash_minus_mu02_dash = mu20_dash - mu02_dash;
		auto const lam_tmp1 = mu20_dash + mu02_dash;
		auto const lam_tmp2 = std::sqrt( 4 * mu11_dash * mu11_dash + mu20_dash_minus_mu02_dash * mu20_dash_minus_mu02_dash );
		auto const lam1_times_two = lam_tmp1 + lam_tmp2, lam2_times_two = lam_tmp1 - lam_tmp2;
		auto const squared_ecc = 1 - lam2_times_two / lam1_times_two;
		if( squared_ecc > 0.98 ) return false;
		x = static_cast< float >( x_bar ), y = static_cast< float >( y_bar );
		sx = std::sqrt( static_cast< float >( mu20 ) );
		sy = std::sqrt( static_cast< float >( mu02 ) );
		return true;
	}
	inline
	bool findMarker(
		cv::Mat &img,
		float const center_x,
		float const center_y,
		float const roi_size_x,
		float const roi_size_y,
		float &new_center_x,
		float &new_center_y,
		float &new_sx,
		float &new_sy,
		unsigned int &keypoint_num,
		char const * const label	// for debug
	)
	{
#if 1
		auto const roi_size_x_half = roi_size_x / 2.0f, roi_size_y_half = roi_size_y / 2.0f;
		cv::Point2f roipt1( center_x - roi_size_x_half, center_y - roi_size_y_half );
		cv::Point2f roipt2( center_x + roi_size_x_half, center_y + roi_size_y_half );

		if( !(0 <= roipt1.x && 0 <= roipt1.y && roipt2.x <= img.cols && roipt2.y <= img.rows) ) return false;

		cv::Mat imgroi = img( cv::Rect( roipt1, roipt2 ) );

		cv::Mat imgroibin, fuck;
		cv::adaptiveThreshold( imgroi, imgroibin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 9, 8 );
		cv::erode( imgroibin, fuck, cv::noArray() );
		// cv::dilate( fuck, fuck, cv::noArray() );

#if FIND_MARKDER_DEBUG_SHOW
		cv::imshow( std::string( "findMarker(" ) + label + ")", fuck );
		cv::waitKey( 1 );
#endif

		std::vector< std::vector< cv::Point2i > > contour_arr;
		cv::findContours( fuck, contour_arr, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		auto const contour_arr_size = std::size( contour_arr );
		auto kpn = 0u;
		auto nx = 0.0f, ny = 0.0f;
		auto nsx = 0.0f, nsy = 0.0f;
		for( auto i = 0u; i < contour_arr_size; ++i )
		{
			float x, y, sx, sy;
			auto const f = key_point_from_contour( x, y, sx, sy, contour_arr[ i ] );
			if( !f ) continue;
			nx += x, ny += y, nsx += sx, nsy += sy;
			++kpn;
		}
		keypoint_num = kpn;
		if( kpn )
		{
			new_center_x = roipt1.x + nx / kpn;
			new_center_y = roipt1.y + ny / kpn;
			new_sx = nsx / kpn;
			new_sy = nsy / kpn;
		}

		return true;
#else
		auto const roi_size_x_half = roi_size_x / 2.0f, roi_size_y_half = roi_size_y / 2.0f;
		cv::Point2f roipt1( center_x - roi_size_x_half, center_y - roi_size_y_half );
		cv::Point2f roipt2( center_x + roi_size_x_half, center_y + roi_size_y_half );

		if( !(0 <= roipt1.x && 0 <= roipt1.y && roipt2.x <= img.cols && roipt2.y <= img.rows) ) return false;

		cv::Mat imgroi = img( cv::Rect( roipt1, roipt2 ) );

		cv::imshow( "imgroi", imgroi );
		cv::Mat imgroibin;
		cv::adaptiveThreshold( imgroi, imgroibin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 7, 8 );
		cv::Mat fuck;
		cv::erode( imgroibin, fuck, cv::noArray() );
		imgroibin = fuck;

		cv::Mat showtmp;
		cv::cvtColor( imgroi, showtmp, cv::COLOR_GRAY2BGR );
		// cv::imshow( "outl", imgroibin );

		auto moment = cv::moments( imgroibin, true );
		auto const x = static_cast< float >( moment.m10 / moment.m00 ), y = static_cast< float >( moment.m01 / moment.m00 );
		if( !(x >= 0 && y >= 0) ) return false;

		std::vector< std::vector< cv::Point2i > > contour_arr;
		cv::findContours( imgroibin, contour_arr, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

		new_center_x = x + roipt1.x;
		new_center_y = y + roipt1.y;
		keypoint_num = static_cast< std::decay_t< decltype( keypoint_num ) > >( contour_arr.size() );

		return true;
#endif
	}
} // namespace