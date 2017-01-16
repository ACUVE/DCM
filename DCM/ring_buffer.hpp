#pragma once

#include <cstddef>
#include <vector>
#include <atomic>

template< typename T >
class ring_buffer
{
private:
	std::vector< T > data;
	std::size_t const data_size;
	std::atomic< std::size_t > index;

public:
	ring_buffer( std::size_t const size, T const &init = T{} )
		: data( size, init )
		, data_size( size )
		, index( 0 )
	{
	}

	std::size_t current_index( std::memory_order const order = std::memory_order_seq_cst ) const
	{
		return index.load( order );
	}
	std::size_t size( void ) const
	{
		return data_size;
	}
	T const &get( std::memory_order const order = std::memory_order_seq_cst ) const
	{
		return get_with_index( index.load( order ) );
	}
	T const &get_with_index( std::size_t const index ) const
	{
		return data[ index % data_size ];
	}

	T &get( std::memory_order const order = std::memory_order_seq_cst )
	{
		return get_with_index( index.load( order ) );
	}
	T &get_next_index( std::memory_order const order = std::memory_order_seq_cst )
	{
		return get_with_index( index.load( order ) + 1 );
	}
	T &get_with_index( std::size_t const index )
	{
		return data[ index % data_size ];
	}
	void next_index( std::memory_order const order = std::memory_order_seq_cst )
	{
		index.fetch_add( 1u, order );
	}
};