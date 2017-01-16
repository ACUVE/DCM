#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <memory>
#include <Fade_2D.h>
#include <unordered_map>
#include <limits>
#include <utility>
#include "meta_programming.hpp"

namespace recognition
{
	struct key_point
	{
		float x, y;
		float sx, sy;
	};
	struct cluster
	{
		float x, y;
		float sx, sy;
		unsigned int num;
	};

	constexpr auto INVALID_INDEX = std::numeric_limits< unsigned int >::max();

	constexpr auto CHECK_NUM = 6u;
	using Cluster_Index = std::tuple< unsigned int, unsigned int >;
	using Cluster_Index_Tuple = meta::multi_tuple_t< CHECK_NUM, Cluster_Index >;
	using Cluster_Num_Tuple = meta::multi_tuple_t< CHECK_NUM, unsigned int >;
	
	struct hash_cnt
	{
		using result_type = std::size_t;
		result_type operator()( Cluster_Num_Tuple const &t ) const
		{
			return calc_tuple< 0u >( t );
		}

	private:
		template< typename INT >
		static
		constexpr
		INT rol3( INT val ){
			static_assert( std::is_unsigned< INT >::value, "Rotate Left only makes sense for unsigned types" );
			return (val << 3) | (val >> (sizeof( INT ) * CHAR_BIT - 3));
		}
		template< std::size_t INDEX >
		static
		std::enable_if_t<
		    (INDEX < std::tuple_size< Cluster_Num_Tuple >::value ),
		    result_type
		> calc_tuple( Cluster_Num_Tuple const &t )
		{
			return
				std::hash< std::tuple_element_t< INDEX, Cluster_Num_Tuple > >()( std::get< INDEX >( t ) ) ^
				rol3( calc_tuple< INDEX + 1 >( t ) );
		}
		template< std::size_t INDEX >
		static
		std::enable_if_t<
		    ( INDEX >= std::tuple_size< Cluster_Num_Tuple >::value ),
		    result_type
		> calc_tuple( Cluster_Num_Tuple const &t )
		{
			return 0u;
		}
	};

	// in recognition.cpp
	void get_num_to_cluster_id( std::vector< std::tuple< std::vector< unsigned int >, std::vector< unsigned int > > > const &cluster_data, std::unordered_map< Cluster_Num_Tuple, Cluster_Index_Tuple, hash_cnt > &map );

	void get_key_point( std::vector< key_point > &ret, cv::Mat &frame_gray, float const area_threshold_min, float const area_threshold_max );
	void get_cluster( std::vector< cluster > &ret, std::vector< key_point > const &kp, float const keypoint_distance_threshold, std::size_t const key_point_grid_num_threshold, unsigned int const max_num_of_key_point_in_cluster, unsigned int const image_width, unsigned int const image_height );
	void get_triangulation( std::vector< GEOM_FADE2D::Triangle2 * > &triangle, std::unique_ptr< GEOM_FADE2D::Fade_2D > &fade2d, std::vector< cluster > const &clu );
	
	void get_cluster_index( std::vector< Cluster_Index > &index, std::unordered_map< Cluster_Num_Tuple, Cluster_Index_Tuple, hash_cnt > const &map, std::vector< cluster > const &clu, std::vector< GEOM_FADE2D::Triangle2 * > const &triangle );

	// in recognition_debug.cpp
	void draw_key_point( cv::Mat &frame_color, std::vector< key_point > const &kp );
	void draw_cluster( cv::Mat &frame_color, std::vector< cluster > const &clu );
	void draw_triangle( cv::Mat &frame_color, std::vector< GEOM_FADE2D::Triangle2 * > const &triangle );

} // namespace recognition
