#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <utility>
#include "tracking_data.hpp"
#include "recognition_thread.hpp"
#include "tracking_thread.hpp"
#include "pnp_track_thread.hpp"
#include "draw_thread.hpp"
#include "ring_buffer.hpp"
#include "capture_thread.hpp"
#include "capture_class.hpp"
#include "ply.hpp"

#ifdef _MSC_VER
#if !_DEBUG
#pragma comment( lib, "opencv_world310" )
#pragma comment( lib, "fade2D_x64_v140_Release" )
#else
#pragma comment( lib, "opencv_world310d" )
#pragma comment( lib, "fade2D_x64_v140_Debug" )
#endif
#endif

template< typename T >
static constexpr
std::add_const_t< T > &as_const( T &t ) noexcept
{
    return t;
}
template< typename T >
static
void as_const( T const && ) = delete;

static
std::vector< cv::Mat > load_images( std::string const &base_name )
{
    std::vector< cv::Mat > vec;
	std::vector< char > path( 255 );
	for( unsigned int i = 0; ; ++i )
	{
		while( true )
		{
			int const num = std::snprintf( &path[ 0 ], path.size(), base_name.c_str(), i );
			if( num < 0 ) break;
			if( static_cast< std::size_t >( num ) < path.size() ) break;
			path.resize( path.size() * 2 );
		}
		cv::Mat img = cv::imread( &path[ 0 ] );
		if( img.empty() )
		{
			if( vec.size() == 0 )
			{
				if( i > 1000 ) break;
			}
			else
			{
				break;
			}
			continue;
		}
		cv::cvtColor( img, img, cv::COLOR_BGR2GRAY );
		vec.emplace_back( std::move( img ) );
	}
	return std::move( vec );
}

static
std::string to_local_path( std::string filepath )
{
#if _WIN32
	std::string::size_type pos = 0u;
	while( true )
	{
		pos = filepath.find( '/', pos );
		if( pos == std::string::npos ) break;
		filepath.replace( pos, 1, 1, '\\');
		pos += 1;
	}
#endif
	return std::move( filepath );
}

int main()
try
{
	using namespace std::string_literals;

	auto const &to_data = R"(../../exp_data/)"s;

    auto const &imgvec = load_images( to_local_path( R"(data/camera_image/human_150/img%03u.bmp)"s ) );
	auto const &filename = to_local_path( to_data + R"(Lau_150_normalized.ply)"s );
	// auto const &filename_hires = R"(..\..\DCM\DCM\data\ply\Laurana50k.ply)"s;
	// std::string filename_num = to_local_path( to_data + R"(Lau_150_normalized.ply.6.1482673125.clu)" );
	std::string filename_num = to_local_path( to_data + R"(lau/150.ply.clu6.1476391063.dat)" );

	auto ply = load_ply( filename );
	auto &point = std::get< 0 >( ply );
	auto &index = std::get< 1 >( ply );
	auto num = load_num( filename_num );

	ximea_xiq_capture xi_cap;

	cv::Mat cameraMatrix = (cv::Mat_< float >( 3, 3 ) <<
		1000.0f, 0.0f, static_cast< float >( xi_cap.width() ) / 2,
		0.0f, 1000.0f, static_cast< float >( xi_cap.height() ) / 2,
		0.0f, 0.0f, 0.0f
	);

	tracking_data td;
	td.point.emplace_back( std::make_unique< point_tracking_data[] >( point.size() / 3 ) );
	td.point_size.push_back( static_cast< unsigned int >( point.size() / 3 ) );
	ring_buffer< cv::Mat > rb( 100 );
	// capture_thread< cv_mat_capture > ct( rb, cv_mat_capture( imgvec ), /*1000u*/ 100u );
	capture_thread< ximea_xiq_capture > ct( rb, std::move( xi_cap ), 0u );
	auto const cluster_data = std::make_tuple( index, num );
	recognition_thread rt( td, as_const( rb ), { cluster_data } );
	tracking_thread tt( td, as_const( rb ), { cluster_data } );
	pnp_track_thread ptt( td, as_const( rb ), { cluster_data }, { point }, cameraMatrix );
	draw_thread dt( td, as_const( rb ), { cluster_data }, { point }, cameraMatrix );
	ct.run();
	rt.run();
	tt.run();
	ptt.run();
	dt.run();
	char c;
	std::cin >> c;
	dt.stop();
	ptt.stop();
	tt.stop();
	rt.stop();
	ct.stop();
}
catch( std::exception &e )
{
	std::cerr << e.what() << std::endl;
	char c;
	std::cin >> c;
}
