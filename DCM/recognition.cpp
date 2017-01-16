#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <array>
#include "recognition.hpp"
#include "findMarker.hpp"

namespace std
{
	template<>
	struct hash< std::tuple< unsigned int, unsigned int > >
	{
		typedef std::tuple< unsigned int, unsigned int > argument_type;
		typedef std::size_t result_type;
		result_type operator()(argument_type const &s) const
		{
			result_type const h1 = std::hash< unsigned int >{}( std::get< 0 >( s ) );
			result_type const h2 = std::hash< unsigned int >{}( std::get< 1 >( s ) );
			return h1 ^ (h2 << 1); // or use boost::hash_combine
		}
	};
}

namespace recognition
{

// ポリゴンの2点のp_indexから残りの1点のp_indexを引く
static
std::unordered_map< std::tuple< unsigned int, unsigned int >, unsigned int > make_vmap( std::vector< unsigned int > const &index )
{
	std::unordered_map< std::tuple< unsigned int, unsigned int >, unsigned int > vmap;
	for( auto i = 0u; i + 2 < std::size( index ); i += 3 )
	{
		auto const ii0 = index[ i + 0 ], ii1 = index[ i + 1 ], ii2 = index[ i + 2 ];
		vmap[ std::make_tuple( ii0, ii1 ) ] = ii2;
		vmap[ std::make_tuple( ii1, ii2 ) ] = ii0;
		vmap[ std::make_tuple( ii2, ii0 ) ] = ii1;
	}
	return std::move( vmap );
}
// ポリゴンの2点のp_indexからそれ自身のindexのを引く
static
std::unordered_map< std::tuple< unsigned int, unsigned int >, unsigned int > make_map_pindex_to_index( std::vector< unsigned int > const &index )
{
	std::unordered_map< std::tuple< unsigned int, unsigned int >, unsigned int > map;
	for( auto i = 0u; i + 2 < std::size( index ); i += 3 )
	{
		auto const ii0 = index[ i + 0 ], ii1 = index[ i + 1 ], ii2 = index[ i + 2 ];
		map[ std::make_tuple( ii0, ii1 ) ] = i;
		map[ std::make_tuple( ii1, ii2 ) ] = i;
		map[ std::make_tuple( ii2, ii0 ) ] = i;
	}
	return std::move( map );
}
void get_num_to_cluster_id( std::vector< std::tuple< std::vector< unsigned int >, std::vector< unsigned int > > > const &cluster_data, std::unordered_map< Cluster_Num_Tuple, Cluster_Index_Tuple, hash_cnt > &map )
{
	for( auto i = 0u; i < std::size( cluster_data ); ++i )
	{
		auto const &t = cluster_data[ i ];
		auto const &index = std::get< 0 >( t );
		auto const &num = std::get< 1 >( t );
		
		auto const &vmap = make_vmap( index );
		// auto const &vmap_index = make_map_pindex_to_index( index );

		auto add_map = [ & ]( auto... arg )
		{
			auto r = map.emplace( std::forward_as_tuple( num[ arg ]... ), std::forward_as_tuple( std::make_tuple( i, arg )... ) );
			if( !r.second ) throw std::logic_error( "get_num_to_cluster_id: same cluster pattern!" );
		};

		auto const index_size = std::size( index );
		for( auto j = 0u; j + 2 < index_size; j += 3 )
		{
			auto const a = index[ j + 0 ], b = index[ j + 1 ], c = index[ j + 2 ];
			auto const u_it = vmap.find( std::make_tuple( b, a ) );
			auto const v_it = vmap.find( std::make_tuple( c, b ) );
			auto const w_it = vmap.find( std::make_tuple( a, c ) );
			if( u_it == vmap.end() || v_it == vmap.end() || w_it == vmap.end() ) continue;
			auto const u = u_it->second, v = v_it->second, w = w_it->second;

			add_map( a, b, c, u, v, w );
			add_map( b, c, a, v, w, u );
			add_map( c, a, b, w, u, v );
		}
	}
}

// この関数，全体的にオーバースペック
void get_key_point( std::vector< key_point > &ret, cv::Mat &frame_gray, float const area_threshold_min, float const area_threshold_max )
{
	// 二値化
	cv::Mat frame_bin;
	cv::adaptiveThreshold( frame_gray, frame_bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 9, 8 );

	cv::Mat fuck;
	cv::erode( frame_bin, fuck, cv::noArray() );
	// cv::dilate( fuck, fuck, cv::noArray() );
	frame_bin = fuck;
	// cv::imshow( "frame_bin", frame_bin );

	// 輪郭抽出（ここ遅そう）
	std::vector< std::vector< cv::Point > > contour_arr;
	// ここが1ms近い
	// cv::findContours( frame_bin, contour_arr, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );
	cv::findContours( frame_bin, contour_arr, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

	cv::Mat dc = frame_bin.clone();
	cv::cvtColor( dc, dc, cv::COLOR_GRAY2BGR );
	cv::drawContours( dc, contour_arr, -1, cv::Scalar( 0, 255, 0 ));
	cv::imshow( "get_key_point", dc );

	auto const contour_arr_size = std::size( contour_arr );
	ret.clear();
	ret.reserve( contour_arr_size );
	for( auto i = 0u; i < contour_arr_size; ++i)
	{
		key_point kp;
		auto const f = key_point_from_contour( kp.x, kp.y, kp.sx, kp.sy, contour_arr[ i ], area_threshold_min, area_threshold_max );
		if( !f ) continue;
		ret.emplace_back( std::move( kp ) );
	}
}

namespace
{	
	struct grid
	{
		std::vector< std::vector< std::size_t > > index;
		unsigned int col, row;
		float width, height;
	};
	
	void devide_keypoint_to_grid( grid &ret, std::vector< key_point > const &kp, float const distance_threshold, std::size_t const key_point_num_threshold, unsigned int const width, unsigned int height )
	{
		auto &index = ret.index;
		for( auto &&v : index ) v.clear();
		unsigned int const grid_col = static_cast< unsigned int >( width / distance_threshold );
		unsigned int const grid_row = static_cast< unsigned int >( height / distance_threshold );
		float const grid_width  = static_cast< float >( width ) / grid_col, grid_height = static_cast< float >( height ) / grid_row;
		ret.col = grid_col, ret.row = grid_row;
		ret.width = grid_width, ret.height = grid_height;
		std::size_t const index_size = grid_col * grid_row, kp_size = kp.size();
		index.resize( index_size );
		for( std::size_t i = 0u; i < kp_size; ++i )
		{
			unsigned int xi = static_cast< unsigned int >( kp[ i ].x / grid_width );
			unsigned int yi = static_cast< unsigned int >( kp[ i ].y / grid_height ); 
			xi = std::min( xi, grid_col - 1 ), yi = std::min( yi, grid_row - 1 );
			index[ yi * grid_col + xi ].emplace_back( i );
		}
		for( auto &&v : index )
		{
			if( v.size() > key_point_num_threshold ) v.clear();
		}
	}
}

struct group
{
	static constexpr std::size_t INVALID_INDEX = std::numeric_limits< std::size_t >::max();
	std::vector< std::size_t > data;
	group( std::size_t size )
		: data( size, INVALID_INDEX )
	{}
	std::size_t get( std::size_t i )
	{
		auto const gi = data[ i ];
		return gi == INVALID_INDEX ? i : data[ i ] = get( gi );
	}
	void same( std::size_t const crr, std::size_t const add )
	{
		auto const ai = get( add );
		auto ci = crr;
		while( true )
		{
			auto const ni = data[ ci ];
			if( ni == ai ) break;
			if( ni == INVALID_INDEX )
			{
				if( ai < ci ) data[ ci ] = ai;
				else data[ ai ] = ci;
				break;
			}
			ci = ni;
		}
	}
};
// 糞遅いコード
void get_cluster( std::vector< cluster > &ret, std::vector< key_point > const &kp, float const keypoint_distance_threshold, std::size_t const key_point_grid_num_threshold, unsigned int const max_num_of_key_point_in_cluster, unsigned int const width, unsigned int height )
{
	ret.clear();

	auto const kp_size = kp.size();
	float const squared_kdt = keypoint_distance_threshold * keypoint_distance_threshold;
	group gi( kp_size );
	for( std::size_t i = 0u; i < kp_size; ++i )
	{
		for( std::size_t j = i + 1; j < kp_size; ++j )
		{
			float const dx = kp[ i ].x - kp[ j ].x, dy = kp[ i ].y - kp[ j ].y;
			if( dx * dx + dy * dy < squared_kdt )
			{
				gi.same( j, i );
			}
		}
	}
	std::vector< std::size_t > index2index( kp_size );
	for( std::size_t i = 0u; i < kp_size; ++i )
	{
		auto const &k = kp[ i ];
		if( gi.data[ i ] == group::INVALID_INDEX )
		{
			cluster c;
			c.x = k.x, c.y = k.y;
			c.sx = k.sx, c.sy = k.sy;
			c.num = 1u;
			index2index[ i ] = ret.size();
			ret.emplace_back( std::move( c ) );
		}
		else
		{
			auto &c = ret[ index2index[ gi.get( i ) ] ];
			c.x += k.x, c.y += k.y;
			c.sx += k.sx, c.sy += k.sy;
			++c.num;
		}
	}
	ret.erase( std::remove_if( ret.begin(), ret.end(), [ & ]( auto const &c ){ return c.num > max_num_of_key_point_in_cluster; } ), ret.end() );
	for( auto &c : ret ) c.x /= c.num, c.y /= c.num, c.sx /= c.num, c.sy /= c.num;

#if 0
	ret.clear();

	std::size_t const kp_size = kp.size();
	float const squared_kdt = keypoint_distance_threshold * keypoint_distance_threshold;

	grid g;
	devide_keypoint_to_grid( g, kp, keypoint_distance_threshold, key_point_grid_num_threshold, width, height );

	std::vector< int > is_added( kp_size, 0 );
	std::vector< std::size_t > cluster_index;	// get_cluster_implで毎回生成するコストを削減するため
	auto get_cluster_impl = [ & ]( std::size_t ind, std::initializer_list< std::size_t > const nums )
	{
		cluster_index.clear();

		for( auto const gind : nums )
		{
			for( auto const kpind : g.index[ gind ] )
			{
				if( is_added[ kpind ] ) continue;
				float const dx = kp[ ind ].x - kp[ kpind ].x, dy = kp[ ind ].y - kp[ kpind ].y;
				if( dx * dx + dy * dy < squared_kdt )
				{
					cluster_index.emplace_back( kpind );
					is_added[ kpind ] = 1;
				}
			}
		}
		if( cluster_index.size() <= max_num_of_key_point_in_cluster )
		{
			auto const cisize = cluster_index.size();
			cluster c;
			c.x = c.y = 0.0f;
			for( auto const cidx : cluster_index )
			{
				auto const &kpcidx = kp[ cidx ];
				c.x += kpcidx.x, c.y += kpcidx.y;
			}
			c.x /= cisize, c.y /= cisize;
			c.num = static_cast< unsigned int >( cisize );
			ret.emplace_back( std::move( c ) );
		}
	};

	for( std::size_t i = 0u; i < kp_size; ++i )
	{
		if( is_added[ i ] ) continue;

		float const xf = kp[ i ].x / g.width, yf = kp[ i ].y / g.height;
		unsigned int xi = static_cast< unsigned int >( xf ), yi = static_cast< unsigned int >( yf );
		xi = std::min( xi, g.col - 1 ), yi = std::min( yi, g.row - 1 );
		unsigned int const ci = xi + yi * g.col;
		bool const up = xf - xi < 0.5, left = yf - yi < 0.5;
		if( up )
		{
			bool const hasup = yi > 0;
			if( left )
			{
				bool const hasleft = xi > 0;
				if( hasup && hasleft ) get_cluster_impl( i, { ci, ci - 1, ci - g.col, ci - g.col - 1 } );
				else if( hasup ) get_cluster_impl( i, { ci, ci - g.col } );
				else if( hasleft ) get_cluster_impl( i, { ci, ci - 1 } );
				else get_cluster_impl( i, { ci } );
			}
			else
			{
				bool const hasright = xi < g.col - 1;
				if( hasup && hasright ) get_cluster_impl( i, { ci, ci + 1, ci - g.col, ci - g.col + 1 } );
				else if( hasup ) get_cluster_impl( i, { ci, g.col } );
				else if( hasright ) get_cluster_impl( i, { ci, ci + 1 } );
				else get_cluster_impl( i, { ci } );
			}
		}
		else
		{
			bool const hasdown = yi < g.row - 1;
			if( hasdown )
			{
				bool const hasleft = xi > 0;
				if( hasdown && hasleft ) get_cluster_impl( i, { ci, ci - 1, ci + g.col, ci + g.col - 1 } );
				else if( hasdown ) get_cluster_impl( i, { ci, ci + g.col} );
				else if( hasleft ) get_cluster_impl( i, { ci, ci - 1 } );
				else get_cluster_impl( i, { ci } );
			}
			else
			{
				bool const hasright = xi < g.col - 1;
				if( hasdown && hasright ) get_cluster_impl( i, { ci, ci + 1, ci + g.col, ci + g.col + 1 } );
				else if( hasdown ) get_cluster_impl( i, { ci , ci + g.col } );
				else if( hasright ) get_cluster_impl( i, { ci, ci + 1 } );
				else get_cluster_impl( i, { ci } );
			}
		}
	}
#endif
}

void get_triangulation( std::vector< GEOM_FADE2D::Triangle2 * > &triangle, std::unique_ptr< GEOM_FADE2D::Fade_2D > &fade2d, std::vector< cluster > const &clu )
{
	auto const clusize = clu.size();
	if( clusize < 3u )
	{
		triangle.clear();
		return;
	}
	fade2d = std::make_unique< GEOM_FADE2D::Fade_2D >( static_cast< unsigned int >( clusize ) );
	std::vector< GEOM_FADE2D::Point2 > vertex_arr;
	vertex_arr.reserve( clusize );
	for( auto i = 0u; i < clusize; ++i )
	{
		// FADE2Dと座標系の取り方が鏡写しのため，yは反転させる
		vertex_arr.emplace_back( clu[ i ].x, -clu[ i ].y );
		vertex_arr.back().setCustomIndex( static_cast< int >( i ) );
	}
	fade2d->insert( vertex_arr );
	fade2d->getTrianglePointers( triangle );
}

void get_cluster_index( std::vector< Cluster_Index > &index, std::unordered_map< Cluster_Num_Tuple, Cluster_Index_Tuple, hash_cnt > const &map, std::vector< cluster > const &clu, std::vector< GEOM_FADE2D::Triangle2 * > const &triangle )
{
#if 1
    auto const clu_size = std::size( clu );
	index.clear();
	index.resize( clu_size, { INVALID_INDEX, 0 } );

	constexpr unsigned int NUM_TOHYO = 12u;
	using Tohyo = std::tuple< std::array< Cluster_Index, NUM_TOHYO + 1 >, unsigned int >;
	std::vector< Tohyo > tohyo( clu_size );
	for( auto const &tri : triangle )
	{
		auto const ot0 = tri->getOppositeTriangle( 0 ), ot1 = tri->getOppositeTriangle( 1 ), ot2 = tri->getOppositeTriangle( 2 );
		if( !ot0 || !ot1 || !ot2 ) continue;
		auto const pt0 = ot0->getIntraTriangleIndex( tri ), pt1 = ot1->getIntraTriangleIndex( tri ), pt2 = ot2->getIntraTriangleIndex( tri );
		unsigned int const a = tri->getCorner( 0 )->getCustomIndex(), b = tri->getCorner( 1 )->getCustomIndex(), c = tri->getCorner( 2 )->getCustomIndex();
		unsigned int const u = ot2->getCorner( pt2 )->getCustomIndex(), v = ot0->getCorner( pt0 )->getCustomIndex(), w = ot1->getCorner( pt1 )->getCustomIndex();
		auto it = [ & ]( auto &&... c ){ return map.find( std::make_tuple( clu[ c ].num... ) ); }( a, b, c, u, v, w );
		if( it == map.end() ) continue;
		meta::for_each( std::forward_as_tuple( a, b, c, u, v, w ), it->second, [ & ]( unsigned int const i, Cluster_Index const &n )
		{
			auto &t = tohyo[ i ];
			auto &a = std::get< 0 >( t );
			auto &s = std::get< 1 >( t );
			if( s < NUM_TOHYO ) a[ s++ ] = n;
			else std::cout << "おかしい" << std::endl;
		} );
	}
	const Cluster_Index INVALID_CLUSTER_INDEX = { INVALID_INDEX, 0 };
	for( auto i = 0u; i < clu_size; ++i )
	{
	    auto &t = tohyo[ i ];
		auto const t_size = std::get< 1 >( t );
		if( t_size < 2 ) continue;
	    auto &t_array = std::get< 0 >( t );
		std::sort( &t_array[ 0 ], &t_array[ t_size ] );
		Cluster_Index const *array_max = nullptr, *array_current = &t_array[ 0 ];
		auto count_max = 0u, count_current = 1u;
		auto update = [ & ]()
		{
			if( count_max == count_current )
			{
				array_max = &INVALID_CLUSTER_INDEX;
			}
			else if( count_max < count_current )
			{
				array_max = array_current;
				count_max = count_current;
			}
		};
		for( auto i = 1u; i < t_size; ++i )
		{
			if( t_array[ i ] == *array_current )
			{
				++count_current;
			}
			else
			{
			    update();
				array_current = &t_array[ i ];
				count_current = 1u;
			}
		}
		update();
		index[ i ] = *array_max;
	}
	// std::cout << "-----------------------------------------------------------" << std::endl;
#elif 0
	index.clear();
	index.resize( clu.size(), { INVALID_INDEX, 0 } );

	constexpr auto INVALID_CNTCIT_INDEX = std::numeric_limits< unsigned int >::max();
	constexpr auto INVALID_BUT_PROCESSED_CNTCIT_INDEX = INVALID_CNTCIT_INDEX - 1u;
	using Clu_Index_Tuple = meta::multi_tuple_t< CHECK_NUM, unsigned int >;
	std::vector< Clu_Index_Tuple > cntcit;
	cntcit.reserve( triangle.size() );
	std::vector< unsigned int > index_updated_cntcit_index( index.size(), INVALID_CNTCIT_INDEX );

	for( auto const &tri : triangle )
	{
		auto const ot0 = tri->getOppositeTriangle( 0 ), ot1 = tri->getOppositeTriangle( 1 ), ot2 = tri->getOppositeTriangle( 2 );
		if( !ot0 || !ot1 || !ot2 ) continue;
		auto const pt0 = ot0->getIntraTriangleIndex( tri ), pt1 = ot1->getIntraTriangleIndex( tri ), pt2 = ot2->getIntraTriangleIndex( tri );
		unsigned int const a = tri->getCorner( 0 )->getCustomIndex(), b = tri->getCorner( 1 )->getCustomIndex(), c = tri->getCorner( 2 )->getCustomIndex();
		unsigned int const u = ot2->getCorner( pt2 )->getCustomIndex(), v = ot0->getCorner( pt0 )->getCustomIndex(), w = ot1->getCorner( pt1 )->getCustomIndex();
		#define ABC a, b, c, u, v, w
		auto it = [ & ]( auto &&... c ){ return map.find( std::make_tuple( clu[ c ].num... ) ); }( ABC );
		if( it == map.end() ) continue;
		bool ok_flag = true;
		meta::for_each( std::forward_as_tuple( ABC ), it->second, [ & ]( auto i, auto n )
		{
			if( !ok_flag ) return;
			auto const v = index_updated_cntcit_index[ i ];
			ok_flag = (v == INVALID_CNTCIT_INDEX) || (v != INVALID_BUT_PROCESSED_CNTCIT_INDEX && index[ v ] == n );
		} );
		if( ok_flag )
		{
			auto const cntcit_index = static_cast< unsigned int >( cntcit.size() );
			cntcit.emplace_back( std::forward_as_tuple( ABC ) );
			meta::for_each( std::forward_as_tuple( ABC ), it->second, [ & ]( auto i, auto n ){
				index[ i ] = n;
				index_updated_cntcit_index[ i ] = cntcit_index;
			} );
		}
		else
		{
			meta::for_each( std::forward_as_tuple( ABC ), [ & ]( auto i ){
				auto const val = index_updated_cntcit_index[ i ];
				if( val == INVALID_CNTCIT_INDEX || val == INVALID_BUT_PROCESSED_CNTCIT_INDEX ) return;
				meta::for_each( cntcit[ val ], [ & ]( auto i )
				{
					index[ i ] = std::make_tuple( INVALID_INDEX, 0 );
					index_updated_cntcit_index[ i ] = INVALID_BUT_PROCESSED_CNTCIT_INDEX;
				} );
			} );
		}
		#undef ABC
	}
#elif 0
	// NGなやつを全部ぶっ殺す前のバージョン
	index.clear();
	index.resize( clu.size(), { INVALID_INDEX, 0 } );

	for( auto const &tri : triangle )
	{
		auto const ot0 = tri->getOppositeTriangle( 0 ), ot1 = tri->getOppositeTriangle( 1 ), ot2 = tri->getOppositeTriangle( 2 );
		if( !ot0 || !ot1 || !ot2 ) continue;
		auto const pt0 = ot0->getIntraTriangleIndex( tri ), pt1 = ot1->getIntraTriangleIndex( tri ), pt2 = ot2->getIntraTriangleIndex( tri );
		unsigned int const a = tri->getCorner( 0 )->getCustomIndex(), b = tri->getCorner( 1 )->getCustomIndex(), c = tri->getCorner( 2 )->getCustomIndex();
		unsigned int const u = ot2->getCorner( pt2 )->getCustomIndex(), v = ot0->getCorner( pt0 )->getCustomIndex(), w = ot1->getCorner( pt1 )->getCustomIndex();
		auto it = [ & ]( auto &&... c ){ return map.find( std::make_tuple( clu[ c ].num... ) ); }( a, b, c, u, v, w );
		if( it == map.end() ) continue;

		// auto ok_num = 0u, ng_num = 0u;
		meta::for_each( std::forward_as_tuple( a, b, c, u, v, w ), it->second, [ & ]( auto i, auto n ){
			if( std::get< 0 >( index[ i ] ) != INVALID_INDEX )
			{
				// if( index[ i ] == n ) ok_num++;
				// else ng_num++;
				if( index[ i ] != n )
				{
					index[ i ] = std::make_tuple( INVALID_INDEX, 0 );
					return;
				}
			}
			index[ i ] = n;
		} );
		// std::clog << "OK: " << ok_num << ", NG: " << ng_num << std::endl;
	}
#endif
}

} // namescape recognition
