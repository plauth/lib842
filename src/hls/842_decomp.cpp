#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>
#define NO_SYNTH

using namespace std;
//#define CHUNK_SIZE 4096

#define OUTPUT_STREAM_WIDTH 64
#define INPUT_STREAM_WIDTH 128
#define BUFFER_WIDTH 266

#define TOKEN_WIDTH 69

/* special templates */
#define OP_REPEAT	(0x1B)
#define OP_ZEROS	(0x1C)
#define OP_END		(0x1E)

/* additional bits of each op param */
#define OP_BITS		(5)
#define REPEAT_BITS	(6)
#define I2_BITS		(8)
#define I4_BITS		(9)
#define I8_BITS		(8)
#define D2_BITS 	(16)
#define D4_BITS 	(32)
#define D8_BITS 	(64)
#define CRC_BITS	(32)
#define N0_BITS		(0)

#define REPEAT_BITS_MAX		(0x3f)

template<uint8_t NB>
struct stream_element {
	ap_uint<8 * NB>       data;
	ap_uint<NB> keep;
	ap_uint<NB> strb;
	ap_uint<1>  last;
};

struct tokenized_stream_element {
	ap_uint<69> data;	// opcode (OP_BITS) + parameters
	ap_uint<1> last;
};

typedef stream_element<INPUT_STREAM_WIDTH/8> input_stream_element;
typedef stream_element<OUTPUT_STREAM_WIDTH/8> output_stream_element;
typedef hls::stream<input_stream_element> input_stream;
typedef hls::stream<output_stream_element> output_stream;

typedef hls::stream<tokenized_stream_element> tokenized_stream;


static inline ap_uint<INPUT_STREAM_WIDTH> swap_endianness_input(ap_uint<INPUT_STREAM_WIDTH> input) {
	ap_uint<INPUT_STREAM_WIDTH> output;

	output(  7,   0) = input(127, 120);
	output( 15,   8) = input(119, 112);
	output( 23,  16) = input(111, 104);
	output( 31,  24) = input(103,  96);
	output( 39,  32) = input( 95,  88);
	output( 47,  40) = input( 87,  80);
	output( 55,  48) = input( 79,  72);
	output( 63,  56) = input( 71,  64);
	output( 71,  64) = input( 63,  56);
	output( 79,  72) = input( 55,  48);
	output( 87,  80) = input( 47,  40);
	output( 95,  88) = input( 39,  32);
	output(103,  96) = input( 31,  24);
	output(111, 104) = input( 23,  16);
	output(119, 112) = input( 15,   8);
	output(127, 120) = input(  7,   0);

	return output;
}

/*
template<uint8_t N> static inline ap_uint<N> read_bits(ulong_stream &in) {
	ap_uint<N> value = in_buffer >> (STREAM_WIDTH - N);

	if (in_buffer_bits < N && !element.last) {
		in >> element;
		in_buffer = swap_endianness(element.data);
	    value |= ap_uint<N>(in_buffer >> (STREAM_WIDTH - (N - in_buffer_bits)));
	    in_buffer <<= N - in_buffer_bits;
		in_buffer_bits += STREAM_WIDTH - N;
		if(in_buffer_bits == 0)
			in_buffer = 0;
	} else {
		in_buffer_bits -= N;
		in_buffer <<= N;
	}

  return value;
}*/

static inline ap_uint<8> get_parameter_length(ap_uint<OP_BITS> op) {
	switch (op) {
		case 0x00: 	// { D8, N0, N0, N0 }, 64 bits
			return D8_BITS + N0_BITS + N0_BITS + N0_BITS;
        case 0x01:	// { D4, D2, I2, N0 }, 56 bits
        	return D4_BITS + D2_BITS + I2_BITS + N0_BITS;
        case 0x02:	// { D4, I2, D2, N0 }, 56 bits
        	return D4_BITS + I2_BITS + D2_BITS + N0_BITS;
		case 0x03: 	// { D4, I2, I2, N0 }, 48 bits
        	return D4_BITS + I2_BITS + I2_BITS + N0_BITS;
		case 0x04:	// { D4, I4, N0, N0 }, 41 bits
        	return D4_BITS + I4_BITS + N0_BITS + N0_BITS;
		case 0x05:	// { D2, I2, D4, N0 }, 56 bits
        	return D2_BITS + I2_BITS + D4_BITS + N0_BITS;
		case 0x06:	// { D2, I2, D2, I2 }, 48 bits
        	return D2_BITS + I2_BITS + D2_BITS + I2_BITS;
		case 0x07:	// { D2, I2, I2, D2 }, 48 bits
        	return D2_BITS + I2_BITS + I2_BITS + D2_BITS;
		case 0x08:	// { D2, I2, I2, I2 }, 40 bits
        	return D2_BITS + I2_BITS + I2_BITS + I2_BITS;
		case 0x09:	// { D2, I2, I4, N0 }, 33 bits
        	return D2_BITS + I2_BITS + I4_BITS + N0_BITS;
		case 0x0a:	// { I2, D2, D4, N0 }, 56 bits
        	return I2_BITS + D2_BITS + D4_BITS + N0_BITS;
		case 0x0b:	// { I2, D4, I2, N0 }, 48 bits
			return I2_BITS + D4_BITS + I2_BITS + N0_BITS;
		case 0x0c:	// { I2, D2, I2, D2 }, 48 bits
			return I2_BITS + D2_BITS + I2_BITS + D2_BITS;
		case 0x0d:	// { I2, D2, I2, I2 }, 40 bits
			return I2_BITS + D2_BITS + I2_BITS + I2_BITS;
		case 0x0e:	// { I2, D2, I4, N0 }, 33 bits
			return I2_BITS + D2_BITS + I4_BITS + N0_BITS;
		case 0x0f:	// { I2, I2, D4, N0 }, 48 bits
			return I2_BITS + I2_BITS + D4_BITS + N0_BITS;
		case 0x10:	// { I2, I2, D2, I2 }, 40 bits
			return I2_BITS + I2_BITS + D2_BITS + I2_BITS;
		case 0x11:	// { I2, I2, I2, D2 }, 40 bits
			return I2_BITS + I2_BITS + I2_BITS + D2_BITS;
		case 0x12:	// { I2, I2, I2, I2 }, 32 bits
			return I2_BITS + I2_BITS + I2_BITS + I2_BITS;
		case 0x13:	// { I2, I2, I4, N0 }, 25 bits
			return I2_BITS + I2_BITS + I4_BITS + N0_BITS;
		case 0x14:	// { I4, D4, N0, N0 }, 41 bits
			return I4_BITS + D4_BITS + N0_BITS + N0_BITS;
		case 0x15:	// { I4, D2, I2, N0 }, 33 bits
			return I4_BITS + D2_BITS + I2_BITS + N0_BITS;
		case 0x16:	// { I4, I2, D2, N0 }, 33 bits
			return I4_BITS + I2_BITS + D2_BITS + N0_BITS;
		case 0x17:	// { I4, I2, I2, N0 }, 25 bits
			return I4_BITS + I2_BITS + I2_BITS + N0_BITS;
		case 0x18:	// { I4, I4, N0, N0 }, 18 bits
			return I4_BITS + I4_BITS + N0_BITS + N0_BITS;
		case 0x19:	// { I8, N0, N0, N0 }, 8 bits
			return I8_BITS + N0_BITS + N0_BITS + N0_BITS;
		case 0x1B:	// OP_REPEAT
			return REPEAT_BITS;
		case 0x1C:  // OP_ZEROS
			return 0;
		case 0x1E:	// OP_END
			return 0;
		default:
			return 0;
	}

	return 0;
}

void opcode_tokenizer(input_stream &in, tokenized_stream &out) {
	ap_uint<BUFFER_WIDTH> input_buffer = 0;
	ap_uint<9> input_buffer_bits = INPUT_STREAM_WIDTH;

	tokenized_stream_element output;
	output.last = 0;
	input_stream_element input = in.read();
	//input.data = swap_endianness_input(input.data);
	input_buffer(BUFFER_WIDTH - 1, BUFFER_WIDTH - INPUT_STREAM_WIDTH) = swap_endianness_input(input.data);

	ap_uint<5> op = OP_END;
	ap_uint<8> payload_length = 0;

	do {
		#pragma HLS PIPELINE
		output.data = 0;

		if(input.last && input_buffer_bits < OP_BITS) {
			input_buffer(BUFFER_WIDTH - 1, BUFFER_WIDTH - OP_BITS) = OP_END;
		}

		op = input_buffer(BUFFER_WIDTH - 1, BUFFER_WIDTH - OP_BITS);
		payload_length = get_parameter_length(op);

		if(input.last && input_buffer_bits < OP_BITS + payload_length) {
			input_buffer(BUFFER_WIDTH - 1, BUFFER_WIDTH - OP_BITS) = op = OP_END;
			payload_length = 0;
		}

		if(op == OP_END) {
			output.last = 1;
		}

		if(!input.last && input_buffer_bits < 2*TOKEN_WIDTH) {
			input = in.read();
			//input.data = swap_endianness_input(input.data);
			input_buffer(BUFFER_WIDTH - input_buffer_bits - 1, BUFFER_WIDTH - INPUT_STREAM_WIDTH - input_buffer_bits) = swap_endianness_input(input.data);
			input_buffer_bits += INPUT_STREAM_WIDTH;
		}

		output.data(TOKEN_WIDTH - 1, TOKEN_WIDTH - OP_BITS - payload_length) = input_buffer(BUFFER_WIDTH - 1, BUFFER_WIDTH - OP_BITS - payload_length);
		input_buffer_bits -= OP_BITS + payload_length;
		input_buffer <<= OP_BITS + payload_length;

		out.write(output);

	} while (!output.last);
}

#ifdef NOSYNTH
void hw842_decompress(ulong_stream &in, ulong_stream &out) {
	#pragma HLS INTERFACE axis port=in
	#pragma HLS INTERFACE axis port=out
	#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

	#pragma HLS DATAFLOW

	tokenized_stream token_stream;
	opcode_tokenizer(in, token_stream);

	ap_uint<8> out_buffer[1024];
	ap_uint<10> out_pos = 0;
	in_buffer = 0;
	in_buffer_bits = 0;

	uint64_t op;
	//ap_uint<REPEAT_BITS> rep;

	//ap_uint<I2_BITS> offset2;
	//ap_uint<I4_BITS> offset4;
	ap_uint<I8_BITS> offset8;
	ap_uint<1> last;

	ap_uint<STREAM_WIDTH> tmp;

	ulong_stream_element out_element;

	do {
		op = read_bits<OP_BITS>(in);

		out_element.strb = 0xFF;
		out_element.keep = 0xFF;
		out_element.last = 0;
		out_element.data = 0;

		switch (op) {
			case 0x00: 	// { D8, N0, N0, N0 }, 64 bits
				out_element.data = read_bits<D8_BITS>(in);
	    	    break;
	    	/*
	        case 0x01:	// { D4, D2, I2, N0 }, 56 bits
	        	do_data<4>(&p);
	        	do_data<2>(&p);
	        	offset2 = read_bits<I2_BITS>(in);
	    	    break;
	        case 0x02:	// { D4, I2, D2, N0 }, 56 bits
	        	do_data<4>(&p);
	        	offset2 = read_bits<I2_BITS>(in);
	        	do_data<2>(&p);
	    	    break;
			case 0x03: 	// { D4, I2, I2, N0 }, 48 bits
	        	do_data<4>(&p);
	        	offset2 = read_bits<I2_BITS>(in);
	        	offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x04:	// { D4, I4, N0, N0 }, 41 bits
	        	do_data<4>(&p);
	        	offset4 = read_bits<I4_BITS>(in);
	        	break;
			case 0x05:	// { D2, I2, D4, N0 }, 56 bits
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
	        	do_data<4>(&p);
	    	    break;
			case 0x06:	// { D2, I2, D2, I2 }, 48 bits
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				break;
			case 0x07:	// { D2, I2, I2, D2 }, 48 bits
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
	    	    break;
			case 0x08:	// { D2, I2, I2, I2 }, 40 bits
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x09:	// { D2, I2, I4, N0 }, 33 bits
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				offset4 = read_bits<I4_BITS>(in);
	        	break;
			case 0x0a:	// { I2, D2, D4, N0 }, 56 bits
				offset2 = read_bits<I2_BITS>(in);
	        	do_data<2>(&p);
	        	do_data<4>(&p);
	    	    break;
			case 0x0b:	// { I2, D4, I2, N0 }, 48 bits
				offset2 = read_bits<I2_BITS>(in);
				do_data<4>(&p);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x0c:	// { I2, D2, I2, D2 }, 48 bits
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
	    	    break;
			case 0x0d:	// { I2, D2, I2, I2 }, 40 bits
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x0e:	// { I2, D2, I4, N0 }, 33 bits
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
				offset4 = read_bits<I4_BITS>(in);
	    	    break;
			case 0x0f:	// { I2, I2, D4, N0 }, 48 bits
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				do_data<4>(&p);
	    	    break;
			case 0x10:	// { I2, I2, D2, I2 }, 40 bits
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x11:	// { I2, I2, I2, D2 }, 40 bits
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
	    	    break;
			case 0x12:	// { I2, I2, I2, I2 }, 32 bits
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x13:	// { I2, I2, I4, N0 }, 25 bits
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				offset4 = read_bits<I4_BITS>(in);
	    	    break;
			case 0x14:	// { I4, D4, N0, N0 }, 41 bits
				offset4 = read_bits<I4_BITS>(in);
				do_data<4>(&p);
	    	    break;
			case 0x15:	// { I4, D2, I2, N0 }, 33 bits
				offset4 = read_bits<I4_BITS>(in);
				do_data<2>(&p);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x16:	// { I4, I2, D2, N0 }, 33 bits
				offset4 = read_bits<I4_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				do_data<2>(&p);
	    	    break;
			case 0x17:	// { I4, I2, I2, N0 }, 25 bits
				offset4 = read_bits<I4_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
			case 0x18:	// { I4, I4, N0, N0 }, 18 bits
				offset2 = read_bits<I2_BITS>(in);
				offset2 = read_bits<I2_BITS>(in);
	    	    break;
	    	*/
			case 0x19:	// { I8, N0, N0, N0 }, 8 bits
				offset8 = read_bits<I8_BITS>(in);
				out_element.data = (out_buffer[(8 * offset8) + 0], out_buffer[(8 * offset8) + 1], out_buffer[(8 * offset8) + 2], out_buffer[(8 * offset8) + 3],
									out_buffer[(8 * offset8) + 4], out_buffer[(8 * offset8) + 5], out_buffer[(8 * offset8) + 6], out_buffer[(8 * offset8) + 7]);
	    	    break;
	    	/*
	    	case OP_REPEAT:
				rep = read_bits(&p, REPEAT_BITS);
				// copy rep + 1
				rep++;

				while (rep-- > 0) {
					memcpy(p.out, p.out - 8, 8);
					p.out += 8;
				}
				break;
			case OP_ZEROS:
				element.data = 0;
				memset(p.out, 0, 8);
				p.out += 8;
				break;
			*/
			case OP_END:
				out_element.last = 1;
				break;
//	        default:
	    }

		out_element.data = swap_endianness(out_element.data);

		out << out_element;
		if(out_pos < 1024 && op != OP_END) {
			out_buffer[out_pos + 0] = out_element.data(63,56);
			out_buffer[out_pos + 1] = out_element.data(55, 48);
			out_buffer[out_pos + 2] = out_element.data(47, 40);
			out_buffer[out_pos + 3] = out_element.data(39, 32);
			out_buffer[out_pos + 4] = out_element.data(31, 24);
			out_buffer[out_pos + 5] = out_element.data(23, 16);
			out_buffer[out_pos + 6] = out_element.data(15, 8 );
			out_buffer[out_pos + 7] = out_element.data(7 , 0 );
			out_pos +=8;
		}
	} while (op != OP_END);

}
#endif
#ifdef NO_SYNTH
#include <inttypes.h>
int main() {
	input_stream streamIn;
	input_stream_element compressed_element;
	compressed_element.strb = 0xFFFF;
	compressed_element.keep = 0xFFFF;


	ap_uint<OUTPUT_STREAM_WIDTH> element0 = 0x9991918989818101;
	ap_uint<OUTPUT_STREAM_WIDTH> element1 = 0x000000000000809F;
	compressed_element.data = (element1, element0);
	compressed_element.last = 1;
	streamIn.write(compressed_element);
	//compressed_element.data = 0x000000000000809F;
	//compressed_element.last = 1;
	//streamIn.write(compressed_element);


	/*
	ulong_stream streamOut;

	hw842_decompress(streamIn, streamOut);

	ulong_stream_element decompressed_element = streamOut.read();
	uint64_t data_uint = decompressed_element.data;
	uint8_t* data = (uint8_t*) malloc(STREAM_WIDTH/8);
	memcpy(data, &data_uint, STREAM_WIDTH/8);

	for (int i = 0; i < STREAM_WIDTH/8; i++) {
		printf("%02x:", data[i]);
	}

	printf("\n\n");

	if(decompressed_element.data == 0x3333323231313030) {
		printf("Test passed\n");
	} else {
		printf("Test failed\n");
	}

	streamOut.read();
	*/

	tokenized_stream streamOut;
	tokenized_stream_element token;

	opcode_tokenizer(streamIn, streamOut);

	do {
		token = streamOut.read();
		ap_uint<OP_BITS> op = token.data(TOKEN_WIDTH - 1, TOKEN_WIDTH - OP_BITS);
		ap_uint<8> parameter_length = get_parameter_length(op);
		ap_uint<OUTPUT_STREAM_WIDTH> parameter = token.data(TOKEN_WIDTH - OP_BITS - 1, 0);
		parameter >>= OUTPUT_STREAM_WIDTH - parameter_length;

		printf("Retrieved Op 0x%02x with parameters 0x%016llx\n", op.to_uchar() ,parameter.to_uint64());
	} while (!token.last);



}
#endif
