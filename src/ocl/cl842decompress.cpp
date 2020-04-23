#include "cl842.h"

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

// Those two symbols are generated during the build process and define
// the source of the decompression OpenCL kernel and the common 842 definitions
extern const char *CL842_DECOMPRESS_842DEFS_SOURCE;
extern const char *CL842_DECOMPRESS_KERNEL_SOURCE;

static constexpr size_t LOCAL_SIZE = 256;

CL842DeviceDecompressor::CL842DeviceDecompressor(const cl::Context &context,
						 const VECTOR_CLASS<cl::Device> &devices,
						 size_t inputChunkSize,
						 size_t inputChunkStride,
						 CL842InputFormat inputFormat,
						 bool verbose)
	: m_inputChunkSize(inputChunkSize),
	  m_inputChunkStride(inputChunkStride),
	  m_inputFormat(inputFormat),
	  m_verbose(verbose)
{
	buildProgram(context, devices);
}

void CL842DeviceDecompressor::decompress(const cl::CommandQueue &commandQueue,
					 const cl::Buffer &inputBuffer,
					 size_t inputOffset, size_t inputSize,
					 const cl::Buffer &inputSizes,
					 const cl::Buffer &outputBuffer,
					 size_t outputOffset, size_t outputSize,
					 const cl::Buffer &outputSizes,
					 const cl::Buffer &returnValues,
					 const VECTOR_CLASS<cl::Event> *events,
					 cl::Event *event)
{
	if (m_inputFormat == CL842InputFormat::INPLACE_COMPRESSED_CHUNKS) {
		if (inputBuffer() != outputBuffer() ||
		    inputOffset != outputOffset || inputSize != outputSize) {
			throw cl::Error(CL_INVALID_VALUE);
		}
	}
	size_t numChunks =
		(inputSize + m_inputChunkStride - 1) / (m_inputChunkStride);

	cl::Kernel decompressKernel(m_program, "decompress");

	decompressKernel.setArg(0, inputBuffer);
	decompressKernel.setArg(1, static_cast<cl_ulong>(inputOffset));
	decompressKernel.setArg(2, inputSizes);
	decompressKernel.setArg(3, outputBuffer);
	decompressKernel.setArg(4, static_cast<cl_ulong>(outputOffset));
	decompressKernel.setArg(5, outputSizes);
	decompressKernel.setArg(6, static_cast<cl_ulong>(numChunks));
	decompressKernel.setArg(7, returnValues);

	cl::NDRange globalSize((numChunks + (LOCAL_SIZE - 1)) &
			       ~(LOCAL_SIZE - 1));
	cl::NDRange localSize(LOCAL_SIZE);

	if (numChunks > 1 && m_verbose) {
		std::cerr << "Using " << numChunks << " chunks of "
			  << m_inputChunkSize << " bytes, " << LOCAL_SIZE
			  << " threads per workgroup" << std::endl;
	}

	std::chrono::steady_clock::time_point t1;
	if (m_verbose) {
		t1 = std::chrono::steady_clock::now();
	}

	commandQueue.enqueueNDRangeKernel(decompressKernel, cl::NullRange,
					  globalSize, localSize, events, event);

	if (m_verbose) {
		commandQueue.finish();

		std::chrono::steady_clock::time_point t2 =
			std::chrono::steady_clock::now();

		auto duration =
			std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
				.count();

		std::cerr
			<< "Decompression performance: " << duration << "ms"
			<< " / "
			<< (static_cast<float>(outputSize) / 1024 / 1024) / (static_cast<float>(duration) / 1000)
			<< "MiB/s" << std::endl;
	}
}

void CL842DeviceDecompressor::buildProgram(
	const cl::Context &context, const VECTOR_CLASS<cl::Device> &devices)
{
	std::ostringstream options;
	options << "-D CL842_CHUNK_SIZE=" << m_inputChunkSize;
	options << " -D CL842_CHUNK_STRIDE=" << m_inputChunkStride;
	options << " -D EINVAL=" << EINVAL;
	options << " -D ENOSPC=" << ENOSPC;
	if (m_inputFormat == CL842InputFormat::MAYBE_COMPRESSED_CHUNKS)
		options << " -D USE_MAYBE_COMPRESSED_CHUNKS=1";
	else if (m_inputFormat == CL842InputFormat::INPLACE_COMPRESSED_CHUNKS)
		options << " -D USE_INPLACE_COMPRESSED_CHUNKS=1";

	std::string src(CL842_DECOMPRESS_KERNEL_SOURCE);
	// Instead of using OpenCL's include mechanism, or duplicating the common 842
	// definitions, we just paste the entire header file on top ourselves
	// This works nicely and avoids us many headaches due to OpenCL headers
	// (most importantly, that for our project, dOpenCL doesn't support them)
	src.insert(0, CL842_DECOMPRESS_842DEFS_SOURCE);
	m_program = cl::Program(context, src);
	try {
		m_program.build(devices, options.str().c_str());
	} catch (const cl::Error &ex) {
		if (ex.err() == CL_BUILD_PROGRAM_FAILURE && m_verbose) {
			std::cerr
				<< "Build Log: "
				<< m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
					   devices[0], nullptr)
				<< std::endl;
		}
		throw;
	}
}

CL842HostDecompressor::CL842HostDecompressor(size_t inputChunkSize,
					     size_t inputChunkStride,
					     CL842InputFormat inputFormat,
					     bool verbose)
	: m_inputChunkStride(inputChunkStride),
	  m_inputFormat(inputFormat),
	  m_verbose(verbose),
	  m_devices(findDevices()), m_context(m_devices),
	  m_queue(m_context, m_devices[0]),
	  m_deviceCompressor(m_context, m_devices, inputChunkSize,
			     inputChunkStride, inputFormat, verbose)
{
}

VECTOR_CLASS<cl::Device> CL842HostDecompressor::findDevices() const
{
	VECTOR_CLASS<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
		if (m_verbose) {
			std::cerr << "ERROR: No OpenCL platforms are available!"
				  << std::endl;
		}
		throw cl::Error(CL_DEVICE_NOT_FOUND);
	}
	if (m_verbose) {
		std::cerr
			<< "Number of available platforms: " << platforms.size()
			<< std::endl;
	}

	VECTOR_CLASS<cl::Device> devices;

	for (auto &deviceType : {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU}) {
		if (m_verbose && deviceType == CL_DEVICE_TYPE_CPU) {
			std::cerr << "WARNING: No OpenCL GPU devices available, "
				  << "falling back to OpenCL CPU devices."
				  << std::endl;
		}
		for (auto &platform : platforms) {
			VECTOR_CLASS<cl::Device> platformDevices;
			try {
				platform.getDevices(deviceType, &platformDevices);
			} catch (cl::Error &ex) {
				if (ex.err() == CL_DEVICE_NOT_FOUND)
					continue;
				throw;
			}

			if (m_verbose) {
				std::cerr << "Platform: "
				  << platform.getInfo<CL_PLATFORM_NAME>()
				  << std::endl;
			}

			for (auto &device : platformDevices) {
				if (!device.getInfo<CL_DEVICE_AVAILABLE>())
					continue;
				if (m_verbose) {
					std::cerr << "Device: "
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

void CL842HostDecompressor::decompress(const uint8_t *input, size_t inputSize,
				       const size_t *inputSizes,
				       uint8_t *output, size_t outputSize,
				       size_t *outputSizes,
				       int *returnValues)
{
	size_t numChunks =
		(inputSize + m_inputChunkStride - 1) / (m_inputChunkStride);

	cl::Buffer inputSizesBuffer;
	if (inputSizes != nullptr) {
		std::vector<cl_ulong> inputSizesCl(inputSizes, inputSizes + numChunks);
		inputSizesBuffer = cl::Buffer(m_context, CL_MEM_READ_ONLY,
					      numChunks * sizeof(cl_ulong));
		m_queue.enqueueWriteBuffer(inputSizesBuffer, CL_TRUE, 0,
					   numChunks * sizeof(cl_ulong),
					   inputSizesCl.data());
	}

	cl::Buffer outputSizesBuffer;
	if (outputSizes != nullptr) {
		std::vector<cl_ulong> outputSizesCl(outputSizes, outputSizes + numChunks);
		outputSizesBuffer = cl::Buffer(m_context, CL_MEM_READ_WRITE,
			numChunks * sizeof(cl_ulong));
		m_queue.enqueueWriteBuffer(outputSizesBuffer, CL_TRUE, 0,
					   numChunks * sizeof(cl_ulong),
					   outputSizesCl.data());
	}

	cl::Buffer returnValuesBuffer;
	if (returnValues != nullptr) {
		returnValuesBuffer = cl::Buffer(m_context, CL_MEM_WRITE_ONLY,
			numChunks * sizeof(cl_int));
	}

	if (m_inputFormat == CL842InputFormat::INPLACE_COMPRESSED_CHUNKS) {
		if (input != output || inputSize != outputSize) {
			throw cl::Error(CL_INVALID_VALUE);
		}

		// Add some more bytes for potential excess lookahead
		cl::Buffer buffer(m_context, CL_MEM_READ_WRITE, inputSize + 64);

		m_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, inputSize,
					   input);
		m_deviceCompressor.decompress(m_queue, buffer, 0,
					      inputSize, inputSizesBuffer,
					      buffer, 0,
					      outputSize, outputSizesBuffer,
					      returnValuesBuffer);
		m_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, outputSize,
					  output);
	} else {
		// Input and output pointers shouldn't overlap (but can't check
		// for that in standard C/C++). Check that they're not the same,
		// but take care about the valid edge case when outputSize == 0
		if (input == output && outputSize != 0) {
			throw cl::Error(CL_INVALID_VALUE);
		}
		if (m_inputFormat ==
			    CL842InputFormat::MAYBE_COMPRESSED_CHUNKS &&
		    inputSize != outputSize) {
			throw cl::Error(CL_INVALID_VALUE);
		}

		cl::Buffer inputBuffer(m_context, CL_MEM_READ_ONLY, inputSize);

		// Avoid a CL_INVALID_BUFFER_SIZE if outputSize == 0
		cl::Buffer outputBuffer(m_context, CL_MEM_READ_WRITE,
					outputSize != 0 ? outputSize : 1);

		m_queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize,
					   input);
		m_deviceCompressor.decompress(m_queue, inputBuffer, 0,
					      inputSize, inputSizesBuffer,
					      outputBuffer, 0,
					      outputSize, outputSizesBuffer,
					      returnValuesBuffer);
		m_queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize,
					  output);
	}

	if (outputSizes != nullptr) {
		std::vector<cl_ulong> outputSizesCl(outputSizes, outputSizes + numChunks);
		m_queue.enqueueReadBuffer(outputSizesBuffer, CL_TRUE, 0,
					  numChunks * sizeof(cl_ulong),
					  outputSizesCl.data());
		std::copy(outputSizesCl.begin(), outputSizesCl.end(), outputSizes);
	}

	if (returnValues != nullptr) {
		static_assert(sizeof(int) == sizeof(cl_int),
			      "sizeof(int) == sizeof(cl_int)");
		m_queue.enqueueReadBuffer(returnValuesBuffer, CL_TRUE, 0,
					  numChunks * sizeof(int),
					  returnValues);
	}
}

int cl842_decompress(const uint8_t *in, size_t ilen,
		     uint8_t *out, size_t *olen)
{
	try {
		static CL842HostDecompressor decompressor(65536, 99999 /* Doesn't matter */,
							 CL842InputFormat::ALWAYS_COMPRESSED_CHUNKS,
							 true);
		int ret;
		decompressor.decompress(in, ilen, &ilen, out, *olen, olen, &ret);

		return ret;
	} catch (const cl::Error &) {
		// Not a great error value, but we shouldn't let C++ exceptions
		// propagate out from the C API
		return -ENOMEM;
	}
}
