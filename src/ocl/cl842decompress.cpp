#include <lib842/cl.h>
#include <lib842/common.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>

#define LIB842_CLDECOMPRESS_USE_PROGRAM_CACHE

#ifdef LIB842_CLDECOMPRESS_USE_PROGRAM_CACHE
#include <cassert>
#include <fstream>
#include "../common/crc32.h"
#endif

// Those two symbols are generated during the build process and define
// the source of the decompression OpenCL kernel and the common 842 definitions
extern const char *LIB842_CLDECOMPRESS_842DEFS_SOURCE;
extern const char *LIB842_CLDECOMPRESS_KERNEL_SOURCE;

namespace lib842 {

static constexpr size_t LOCAL_SIZE = 256;

CLDeviceDecompressor::CLDeviceDecompressor(const cl::Context &context,
					   const cl::vector<cl::Device> &devices,
					   size_t chunkSize,
					   size_t chunkStride,
					   CLDecompressorInputFormat inputFormat,
					   std::function<std::ostream&(void)> error_logger,
					   std::function<std::ostream&(void)> debug_logger)
	: m_chunkSize(chunkSize),
	  m_chunkStride(chunkStride),
	  m_inputFormat(inputFormat),
	  m_error_logger(std::move(error_logger)),
	  m_debug_logger(std::move(debug_logger))
{
	buildProgram(context, devices);
}

void CLDeviceDecompressor::decompress(const cl::CommandQueue &commandQueue,
				      size_t numChunks,
				      const cl::Buffer &inputBuffer,
				      size_t inputOffset,
				      const cl::Buffer &inputChunkSizes,
				      const cl::Buffer &outputBuffer,
				      size_t outputOffset,
				      const cl::Buffer &outputChunkSizes,
				      const cl::Buffer &chunkShuffleMap,
				      const cl::Buffer &returnValues,
				      const cl::vector<cl::Event> *events,
				      cl::Event *event) const
{
	if (m_inputFormat == CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS &&
	    (inputBuffer() != outputBuffer() || inputOffset != outputOffset)) {
		throw cl::Error(CL_INVALID_VALUE);
	}

	cl::Kernel decompressKernel(m_program, "decompress");

	decompressKernel.setArg(0, inputBuffer);
	decompressKernel.setArg(1, static_cast<cl_ulong>(inputOffset));
	decompressKernel.setArg(2, inputChunkSizes);
	decompressKernel.setArg(3, outputBuffer);
	decompressKernel.setArg(4, static_cast<cl_ulong>(outputOffset));
	decompressKernel.setArg(5, outputChunkSizes);
	decompressKernel.setArg(6, static_cast<cl_ulong>(numChunks));
	decompressKernel.setArg(7, chunkShuffleMap);
	decompressKernel.setArg(8, returnValues);

	cl::NDRange globalSize((numChunks + (LOCAL_SIZE - 1)) &
			       ~(LOCAL_SIZE - 1));
	cl::NDRange localSize(LOCAL_SIZE);

	commandQueue.enqueueNDRangeKernel(decompressKernel, cl::NullRange,
					  globalSize, localSize, events, event);
}

#ifdef LIB842_CLDECOMPRESS_USE_PROGRAM_CACHE
/* In OpenCL, for portability, programs are normally passed as strings and
 * built at run-time for the specific device they are run on, which is
 * expensive, specially if running many small applications (e.g. unit tests).
 *
 * Unfortunately, OpenCL does not require implementations to provide a
 * program cache (as of 2020-05-09, NVIDIA does, Intel and AMD don't),
 * however, it provides the means for applications to implement it themselves.
 *
 * This is a very simple cache for the OpenCL decompression kernel.
 */
struct program_cache
{
	// Hash of the program source
	uint32_t sourceHash;
	// Program build options (#define's)
	size_t chunkSize;
	size_t chunkStride;
	CLDecompressorInputFormat inputFormat;
	// Information of the devices the program was built for
	// TODO: Is this enough to ensure we don't accidentally use the wrong cache?
	cl::vector<cl::string> deviceNames;

	cl::Program::Binaries find() const {
		std::ifstream in(CACHE_PATH, std::ifstream::in | std::ifstream::binary);
		in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

		program_cache disk_cache;
		disk_cache.readMetadata(in);
		if (chunkSize != disk_cache.chunkSize ||
		    chunkStride != disk_cache.chunkStride ||
		    inputFormat != disk_cache.inputFormat ||
		    sourceHash != disk_cache.sourceHash ||
		    deviceNames != disk_cache.deviceNames) {
			return {};
		}

		return readBinaries(in, deviceNames.size());
	}

	void set(const cl::Program::Binaries &binaries) const {
		assert(deviceNames.size() == binaries.size());

		std::ofstream out(CACHE_PATH, std::ofstream::out | std::ofstream::binary);
		out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
		writeMetadata(out);
		writeBinaries(out, binaries);
	}

private:
	static constexpr const char *CACHE_PATH = "/tmp/cl842_cache.bin";

	void readMetadata(std::ifstream &in) {
		in.read(reinterpret_cast<char *>(&chunkSize), sizeof(chunkSize));
		in.read(reinterpret_cast<char *>(&chunkStride), sizeof(chunkStride));
		in.read(reinterpret_cast<char *>(&inputFormat), sizeof(inputFormat));
		in.read(reinterpret_cast<char *>(&sourceHash), sizeof(sourceHash));

		size_t numDevices;
		in.read(reinterpret_cast<char *>(&numDevices), sizeof(numDevices));

		for (size_t i = 0; i < numDevices; i++) {
			size_t length;
			in.read(reinterpret_cast<char *>(&length), sizeof(length));
			cl::string deviceName(length, '\0');
			in.read(&deviceName[0], deviceName.size());
			deviceNames.push_back(deviceName);
		}
	}

	static cl::Program::Binaries readBinaries(std::ifstream &in, size_t numDevices) {
		cl::Program::Binaries binaries;
		for (size_t i = 0; i < numDevices; i++) {
			size_t size;
			in.read(reinterpret_cast<char *>(&size), sizeof(size));
			cl::vector<uint8_t> binary(size);
			in.read(reinterpret_cast<char *>(binary.data()), binary.size());
			binaries.push_back(binary);
		}
		return binaries;
	}

	void writeMetadata(std::ofstream &out) const {
		out.write(reinterpret_cast<const char *>(&chunkSize), sizeof(chunkSize));
		out.write(reinterpret_cast<const char *>(&chunkStride), sizeof(chunkStride));
		out.write(reinterpret_cast<const char *>(&inputFormat), sizeof(inputFormat));
		out.write(reinterpret_cast<const char *>(&sourceHash), sizeof(sourceHash));

		size_t numDevices = deviceNames.size();
		out.write(reinterpret_cast<const char *>(&numDevices), sizeof(numDevices));

		for (const auto &dn : deviceNames) {
			size_t length = dn.length();
			out.write(reinterpret_cast<const char *>(&length), sizeof(length));
			out.write(dn.data(), dn.size());
		}
	}

	static void writeBinaries(std::ofstream &out, const cl::Program::Binaries &binaries) {
		for (const auto &b : binaries) {
			size_t size = b.size();
			out.write(reinterpret_cast<const char *>(&size), sizeof(size));
			out.write(reinterpret_cast<const char *>(b.data()), b.size());
		}
	}
};
#endif

void CLDeviceDecompressor::buildProgram(
	const cl::Context &context, const cl::vector<cl::Device> &devices)
{
	cl::string src(LIB842_CLDECOMPRESS_KERNEL_SOURCE);
	// Instead of using OpenCL's include mechanism, or duplicating the common 842
	// definitions, we just paste the entire header file on top ourselves
	// This works nicely and avoids us many headaches due to OpenCL headers
	// (most importantly, that for our project, dOpenCL doesn't support them)
	src.insert(0, LIB842_CLDECOMPRESS_842DEFS_SOURCE);

	std::ostringstream options;
	options << "-D CL842_CHUNK_SIZE=" << m_chunkSize;
	options << " -D CL842_CHUNK_STRIDE=" << m_chunkStride;
	options << " -D EINVAL=" << EINVAL;
	options << " -D ENOSPC=" << ENOSPC;
	if (m_inputFormat == CLDecompressorInputFormat::MAYBE_COMPRESSED_CHUNKS ||
	    m_inputFormat == CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS) {
		options << " -D LIB842_COMPRESSED_CHUNK_MARKER_DEF={";
		std::copy(std::begin(LIB842_COMPRESSED_CHUNK_MARKER),
			  std::end(LIB842_COMPRESSED_CHUNK_MARKER),
			  std::ostream_iterator<double>(options, ","));
		options << "}";
	}
	if (m_inputFormat == CLDecompressorInputFormat::MAYBE_COMPRESSED_CHUNKS)
		options << " -D USE_MAYBE_COMPRESSED_CHUNKS=1";
	else if (m_inputFormat == CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS)
		options << " -D USE_INPLACE_COMPRESSED_CHUNKS=1";

#ifdef LIB842_CLDECOMPRESS_USE_PROGRAM_CACHE
	program_cache cache;
	cache.sourceHash = crc32_be(0, reinterpret_cast<const uint8_t *>(src.data()), src.length());
	cache.chunkSize = m_chunkSize;
	cache.chunkStride = m_chunkStride;
	cache.inputFormat = m_inputFormat;
	for (const auto &d : devices)
		cache.deviceNames.push_back(d.getInfo<CL_DEVICE_NAME>());

	cl::Program::Binaries binaries;
	try {
		binaries = cache.find();
	} catch (const std::ifstream::failure &) {
		m_debug_logger()
			<< "Could not read lib842's OpenCL program cache, regenerating"
			<< std::endl;
	}
	if (!binaries.empty()) {
		try {
			m_program = cl::Program(context, devices, binaries);
			m_program.build(devices, options.str().c_str());
			return;
		} catch (const cl::Error &ex) {
			m_debug_logger()
				<< "Building the lib842's OpenCL program"
				<< " from cache failed, rebuilding from source"
				<< std::endl;
		}
	}
#endif

	m_program = cl::Program(context, src);
	try {
		m_program.build(devices, options.str().c_str());
	} catch (const cl::Error &ex) {
		if (ex.err() == CL_BUILD_PROGRAM_FAILURE) {
			m_error_logger()
				<< "Building the lib842's OpenCL program"
				<< " from source failed, build log is:\n"
				<< m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
					   devices[0], nullptr)
				<< std::endl;
		}
		throw;
	}
#ifdef LIB842_CLDECOMPRESS_USE_PROGRAM_CACHE
	cache.set(m_program.getInfo<CL_PROGRAM_BINARIES>());
#endif
}

CLHostDecompressor::CLHostDecompressor(size_t chunkSize,
				       size_t chunkStride,
				       CLDecompressorInputFormat inputFormat,
				       bool verbose, bool profile)
	: m_chunkSize(chunkSize),
	  m_chunkStride(chunkStride),
	  m_inputFormat(inputFormat),
	  m_verbose(verbose), m_profile(profile),
	  m_devices(findDevices()), m_context(m_devices),
	  m_queue(m_context, m_devices[0], profile ? CL_QUEUE_PROFILING_ENABLE : 0),
	  m_deviceCompressor(m_context, m_devices, chunkSize,
			     chunkStride, inputFormat,
			     []()  -> std::ostream& { return std::cerr; },
			     []()  -> std::ostream& { return std::cout; })
{
}

cl::vector<cl::Device> CLHostDecompressor::findDevices() const
{
	cl::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
		if (m_verbose) {
			std::cerr << "ERROR: No OpenCL platforms are available!"
				  << std::endl;
		}
		throw cl::Error(CL_DEVICE_NOT_FOUND);
	}
	if (m_verbose) {
		std::cout
			<< "Number of available platforms: " << platforms.size()
			<< std::endl;
	}

	cl::vector<cl::Device> devices;

	for (auto &deviceType : {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU}) {
		if (m_verbose && deviceType == CL_DEVICE_TYPE_CPU) {
			std::cout << "WARNING: No OpenCL GPU devices available, "
				  << "falling back to OpenCL CPU devices."
				  << std::endl;
		}
		for (auto &platform : platforms) {
			cl::vector<cl::Device> platformDevices;
			try {
				platform.getDevices(deviceType, &platformDevices);
			} catch (cl::Error &ex) {
				if (ex.err() == CL_DEVICE_NOT_FOUND)
					continue;
				throw;
			}

			if (m_verbose) {
				std::cout << "Platform: "
				  << platform.getInfo<CL_PLATFORM_NAME>()
				  << std::endl;
			}

			for (auto &device : platformDevices) {
				if (!device.getInfo<CL_DEVICE_AVAILABLE>())
					continue;
				if (m_verbose) {
					std::cout << "Device: "
						  << device.getInfo<CL_DEVICE_NAME>()
						  << std::endl;
				}
				devices.push_back(device);
			}
			if (!devices.empty())
				return devices;
		}
	}

	if (m_verbose) {
		std::cerr << "ERROR: No OpenCL devices are available!"
			  << std::endl;
	}
	throw cl::Error(CL_DEVICE_NOT_FOUND);
}

void CLHostDecompressor::decompress(const uint8_t *input, size_t inputBufferSize,
				    const size_t *inputChunkSizes,
				    uint8_t *output, size_t outputBufferSize,
				    size_t *outputChunkSizes,
				    size_t *chunkShuffleMap,
				    int *returnValues,
				    long long *time) const
{
	static constexpr bool CLBUF_READ_ONLY = true, CLBUF_READ_WRITE = false;

	size_t numChunks = (inputBufferSize + m_chunkStride - 1) / m_chunkStride;

	cl::Buffer inputSizesBuffer;
	if (inputChunkSizes != nullptr) {
		std::vector<cl_ulong> inputSizesCl(inputChunkSizes, inputChunkSizes + numChunks);
		inputSizesBuffer = cl::Buffer(m_queue,
			inputSizesCl.begin(), inputSizesCl.end(), CLBUF_READ_ONLY);
	}

	cl::Buffer outputSizesBuffer;
	if (outputChunkSizes != nullptr) {
		std::vector<cl_ulong> outputSizesCl(outputChunkSizes, outputChunkSizes + numChunks);
		outputSizesBuffer = cl::Buffer(m_queue,
			outputSizesCl.begin(), outputSizesCl.end(), CLBUF_READ_WRITE);
	}

	cl::Buffer chunkShuffleMapBuffer;
	if (chunkShuffleMap != nullptr) {
		std::vector<cl_ulong> chunkShuffleMapCl(chunkShuffleMap, chunkShuffleMap + numChunks);
		chunkShuffleMapBuffer = cl::Buffer(m_queue,
			chunkShuffleMapCl.begin(), chunkShuffleMapCl.end(), CLBUF_READ_ONLY);
	}

	cl::Buffer returnValuesBuffer;
	if (returnValues != nullptr) {
		returnValuesBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY,
			numChunks * sizeof(cl_int));
	}

	if (numChunks > 1 && m_verbose) {
		std::cout << "Using " << numChunks << " chunks of "
			  << m_chunkSize << " bytes, " << LOCAL_SIZE
			  << " threads per workgroup" << std::endl;
	}

	cl::Event decompressEvent;

	if (m_inputFormat == CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS) {
		if (input != output || inputBufferSize != outputBufferSize) {
			throw cl::Error(CL_INVALID_VALUE);
		}

		// Add some more bytes for potential excess lookahead
		cl::Buffer buffer(m_context, CL_MEM_READ_WRITE, inputBufferSize + 64);

		cl::copy(m_queue, input, input + inputBufferSize, buffer);
		m_deviceCompressor.decompress(m_queue, numChunks,
					      buffer, 0, inputSizesBuffer,
					      buffer, 0, outputSizesBuffer,
					      chunkShuffleMapBuffer, returnValuesBuffer,
					      nullptr, &decompressEvent);
		decompressEvent.wait();
		cl::copy(m_queue, buffer, output, output + outputBufferSize);
	} else {
		// Input and output pointers shouldn't overlap (but can't check
		// for that in standard C/C++). Check that they're not the same,
		// but take care about the valid edge case when outputBufferSize == 0
		if (input == output && outputBufferSize != 0) {
			throw cl::Error(CL_INVALID_VALUE);
		}
		if (m_inputFormat ==
			    CLDecompressorInputFormat::MAYBE_COMPRESSED_CHUNKS &&
		    inputBufferSize != outputBufferSize) {
			throw cl::Error(CL_INVALID_VALUE);
		}

		cl::Buffer inputBuffer(m_context, CL_MEM_READ_ONLY, inputBufferSize);
		// Avoid a CL_INVALID_BUFFER_SIZE if outputBufferSize == 0
		cl::Buffer outputBuffer(m_context, CL_MEM_READ_WRITE,
					outputBufferSize != 0 ? outputBufferSize : 1);

		cl::copy(m_queue, input, input + inputBufferSize, inputBuffer);
		m_deviceCompressor.decompress(m_queue, numChunks,
					      inputBuffer, 0, inputSizesBuffer,
					      outputBuffer, 0, outputSizesBuffer,
					      chunkShuffleMapBuffer, returnValuesBuffer,
					      nullptr, &decompressEvent);
		decompressEvent.wait();
		cl::copy(m_queue, outputBuffer, output, output + outputBufferSize);
	}

	if (m_profile && time != nullptr) {
		auto timeStartNs = decompressEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		auto timeEndNs = decompressEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		*time = static_cast<long long>((timeEndNs - timeStartNs) / 1000000);
	} else if (time != nullptr) {
		*time = -1;
	}

	if (outputChunkSizes != nullptr) {
		std::vector<cl_ulong> outputSizesCl(outputChunkSizes, outputChunkSizes + numChunks);
		cl::copy(m_queue, outputSizesBuffer, outputSizesCl.begin(), outputSizesCl.end());
		std::copy(outputSizesCl.begin(), outputSizesCl.end(), outputChunkSizes);
	}

	if (returnValues != nullptr) {
		static_assert(sizeof(int) == sizeof(cl_int),
			      "sizeof(int) == sizeof(cl_int)");
		cl::copy(m_queue, returnValuesBuffer, returnValues, returnValues + numChunks);
	}
}

} // namespace lib842

int cl842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen)
{
	try {
		static lib842::CLHostDecompressor decompressor(
			65536, 99999 /* Doesn't matter */,
			lib842::CLDecompressorInputFormat::ALWAYS_COMPRESSED_CHUNKS,
			true, false);
		int ret;
		decompressor.decompress(in, ilen, &ilen, out, *olen, olen, nullptr, &ret, nullptr);
		return ret;
	} catch (const cl::Error &) {
		// Not a great error value, but we shouldn't let C++ exceptions
		// propagate out from the C API
		return -ENOMEM;
	}
}
