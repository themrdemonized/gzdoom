/*
** gl_hqresize.cpp
** Contains high quality upsampling functions.
** So far Scale2x/3x/4x as described in http://scale2x.sourceforge.net/
** are implemented.
**
**---------------------------------------------------------------------------
** Copyright 2008 Benjamin Berkels
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions
** are met:
**
** 1. Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
** 2. Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in the
**    documentation and/or other materials provided with the distribution.
** 3. The name of the author may not be used to endorse or promote products
**    derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
** IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
** OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
** IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
** NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
** THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**---------------------------------------------------------------------------
**
*/

#include "c_cvars.h"
#include "hqnx/hqx.h"
#ifdef HAVE_MMX
#include "hqnx_asm/hqnx_asm.h"
#endif
#include <memory>
#include "xbr/xbrz.h"
#include "xbr/xbrz_old.h"
#include "parallel_for.h"
#include "textures.h"
#include "texturemanager.h"
#include "printf.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

int upscalemask;

EXTERN_CVAR(Int, gl_texture_hqresizemult)
CUSTOM_CVAR(Int, gl_texture_hqresizemode, 0, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL)
{
	if (self < 0 || self > 6)
		self = 0;
	if ((gl_texture_hqresizemult > 4) && (self < 4) && (self > 0))
		gl_texture_hqresizemult = 4;
	TexMan.FlushAll();
	UpdateUpscaleMask();
}

CUSTOM_CVAR(Int, gl_texture_hqresizemult, 1, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL)
{
	if (self < 1 || self > 6)
		self = 1;
	if ((self > 4) && (gl_texture_hqresizemode < 4) && (gl_texture_hqresizemode > 0))
		self = 4;
	TexMan.FlushAll();
	UpdateUpscaleMask();
}

CUSTOM_CVAR(Int, gl_texture_hqresize_maxinputsize, 512, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL)
{
	if (self > 1024) self = 1024;
	TexMan.FlushAll();
}

CUSTOM_CVAR(Int, gl_texture_hqresize_targets, 15, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL)
{
	TexMan.FlushAll();
	UpdateUpscaleMask();
}

CVAR (Flag, gl_texture_hqresize_textures, gl_texture_hqresize_targets, 1);
CVAR (Flag, gl_texture_hqresize_sprites, gl_texture_hqresize_targets, 2);
CVAR (Flag, gl_texture_hqresize_fonts, gl_texture_hqresize_targets, 4);
CVAR (Flag, gl_texture_hqresize_skins, gl_texture_hqresize_targets, 8);

CVAR(Bool, gl_texture_hqresize_multithread, true, CVAR_ARCHIVE | CVAR_GLOBALCONFIG);

CUSTOM_CVAR(Int, gl_texture_hqresize_mt_width, 16, CVAR_ARCHIVE | CVAR_GLOBALCONFIG)
{
	if (self < 2)    self = 2;
	if (self > 1024) self = 1024;
}

CUSTOM_CVAR(Int, gl_texture_hqresize_mt_height, 4, CVAR_ARCHIVE | CVAR_GLOBALCONFIG)
{
	if (self < 2)    self = 2;
	if (self > 1024) self = 1024;
}

CVAR(Int, xbrz_colorformat, 0, CVAR_ARCHIVE | CVAR_GLOBALCONFIG)

void UpdateUpscaleMask()
{
	if (!gl_texture_hqresizemode || gl_texture_hqresizemult == 1) upscalemask = 0;
	else upscalemask = gl_texture_hqresize_targets;
}


static void xbrzApplyOptions()
{
	if (gl_texture_hqresizemult != 0 && (gl_texture_hqresizemode == 4 || gl_texture_hqresizemode == 5))
	{
		if (xbrz_colorformat == 0)
		{
			Printf("Changing xBRZ options requires a restart when buffered color format is used.\n"
				"To avoid this at cost of scaling performance, set xbrz_colorformat CVAR to non-zero value.");
		}
		else
		{
			TexMan.FlushAll();
		}
	}
}

#define XBRZ_CVAR(NAME, VALUE) \
	CUSTOM_CVAR(Float, xbrz_##NAME, VALUE, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL) { xbrzApplyOptions(); }

XBRZ_CVAR(luminanceweight, 1.f)
XBRZ_CVAR(equalcolortolerance, 30.f)
XBRZ_CVAR(centerdirectionbias, 4.f)
XBRZ_CVAR(dominantdirectionthreshold, 3.6f)
XBRZ_CVAR(steepdirectionthreshold, 2.2f)

#undef XBRZ_CVAR

static void scale2x ( uint32_t* inputBuffer, uint32_t* outputBuffer, int inWidth, int inHeight )
{
	const int width = 2* inWidth;
	const int height = 2 * inHeight;

	for ( int i = 0; i < inWidth; ++i )
	{
		const int iMinus = (i > 0) ? (i-1) : 0;
		const int iPlus = (i < inWidth - 1 ) ? (i+1) : i;
		for ( int j = 0; j < inHeight; ++j )
		{
			const int jMinus = (j > 0) ? (j-1) : 0;
			const int jPlus = (j < inHeight - 1 ) ? (j+1) : j;
			const uint32_t A = inputBuffer[ iMinus +inWidth*jMinus];
			const uint32_t B = inputBuffer[ iMinus +inWidth*j    ];
			const uint32_t C = inputBuffer[ iMinus +inWidth*jPlus];
			const uint32_t D = inputBuffer[ i     +inWidth*jMinus];
			const uint32_t E = inputBuffer[ i     +inWidth*j    ];
			const uint32_t F = inputBuffer[ i     +inWidth*jPlus];
			const uint32_t G = inputBuffer[ iPlus +inWidth*jMinus];
			const uint32_t H = inputBuffer[ iPlus +inWidth*j    ];
			const uint32_t I = inputBuffer[ iPlus +inWidth*jPlus];
			if (B != H && D != F) {
				outputBuffer[2*i   + width*2*j    ] = D == B ? D : E;
				outputBuffer[2*i   + width*(2*j+1)] = B == F ? F : E;
				outputBuffer[2*i+1 + width*2*j    ] = D == H ? D : E;
				outputBuffer[2*i+1 + width*(2*j+1)] = H == F ? F : E;
			} else {
				outputBuffer[2*i   + width*2*j    ] = E;
				outputBuffer[2*i   + width*(2*j+1)] = E;
				outputBuffer[2*i+1 + width*2*j    ] = E;
				outputBuffer[2*i+1 + width*(2*j+1)] = E;
			}
		}
	}
}

static void scale3x ( uint32_t* inputBuffer, uint32_t* outputBuffer, int inWidth, int inHeight )
{
	const int width = 3* inWidth;
	const int height = 3 * inHeight;

	for ( int i = 0; i < inWidth; ++i )
	{
		const int iMinus = (i > 0) ? (i-1) : 0;
		const int iPlus = (i < inWidth - 1 ) ? (i+1) : i;
		for ( int j = 0; j < inHeight; ++j )
		{
			const int jMinus = (j > 0) ? (j-1) : 0;
			const int jPlus = (j < inHeight - 1 ) ? (j+1) : j;
			const uint32_t A = inputBuffer[ iMinus +inWidth*jMinus];
			const uint32_t B = inputBuffer[ iMinus +inWidth*j    ];
			const uint32_t C = inputBuffer[ iMinus +inWidth*jPlus];
			const uint32_t D = inputBuffer[ i     +inWidth*jMinus];
			const uint32_t E = inputBuffer[ i     +inWidth*j    ];
			const uint32_t F = inputBuffer[ i     +inWidth*jPlus];
			const uint32_t G = inputBuffer[ iPlus +inWidth*jMinus];
			const uint32_t H = inputBuffer[ iPlus +inWidth*j    ];
			const uint32_t I = inputBuffer[ iPlus +inWidth*jPlus];
			if (B != H && D != F) {
				outputBuffer[3*i   + width*3*j    ] = D == B ? D : E;
				outputBuffer[3*i   + width*(3*j+1)] = (D == B && E != C) || (B == F && E != A) ? B : E;
				outputBuffer[3*i   + width*(3*j+2)] = B == F ? F : E;
				outputBuffer[3*i+1 + width*3*j    ] = (D == B && E != G) || (D == H && E != A) ? D : E;
				outputBuffer[3*i+1 + width*(3*j+1)] = E;
				outputBuffer[3*i+1 + width*(3*j+2)] = (B == F && E != I) || (H == F && E != C) ? F : E;
				outputBuffer[3*i+2 + width*3*j    ] = D == H ? D : E;
				outputBuffer[3*i+2 + width*(3*j+1)] = (D == H && E != I) || (H == F && E != G) ? H : E;
				outputBuffer[3*i+2 + width*(3*j+2)] = H == F ? F : E;
			} else {
				outputBuffer[3*i   + width*3*j    ] = E;
				outputBuffer[3*i   + width*(3*j+1)] = E;
				outputBuffer[3*i   + width*(3*j+2)] = E;
				outputBuffer[3*i+1 + width*3*j    ] = E;
				outputBuffer[3*i+1 + width*(3*j+1)] = E;
				outputBuffer[3*i+1 + width*(3*j+2)] = E;
				outputBuffer[3*i+2 + width*3*j    ] = E;
				outputBuffer[3*i+2 + width*(3*j+1)] = E;
				outputBuffer[3*i+2 + width*(3*j+2)] = E;
			}
		}
	}
}

static void scale4x ( uint32_t* inputBuffer, uint32_t* outputBuffer, int inWidth, int inHeight )
{
	int width = 2* inWidth;
	int height = 2 * inHeight;
	uint32_t * buffer2x = new uint32_t[width*height];

	scale2x ( reinterpret_cast<uint32_t*> ( inputBuffer ), reinterpret_cast<uint32_t*> ( buffer2x ), inWidth, inHeight );
	width *= 2;
	height *= 2;
	scale2x ( reinterpret_cast<uint32_t*> ( buffer2x ), reinterpret_cast<uint32_t*> ( outputBuffer ), 2*inWidth, 2*inHeight );
	delete[] buffer2x;
}

static unsigned char *scaleNxHelper( void (*scaleNxFunction) ( uint32_t* , uint32_t* , int , int),
							  const int N,
							  unsigned char *inputBuffer,
							  const int inWidth,
							  const int inHeight,
							  int &outWidth,
							  int &outHeight )
{
	outWidth = N * inWidth;
	outHeight = N *inHeight;
	unsigned char * newBuffer = new unsigned char[outWidth*outHeight*4];

	scaleNxFunction ( reinterpret_cast<uint32_t*> ( inputBuffer ), reinterpret_cast<uint32_t*> ( newBuffer ), inWidth, inHeight );
	delete[] inputBuffer;
	return newBuffer;
}

static unsigned char *normalNx(const int N,
							  unsigned char *inputBuffer,
							  const int inWidth,
							  const int inHeight,
							  int &outWidth,
							  int &outHeight )
{
	outWidth = N * inWidth;
	outHeight = N *inHeight;
	unsigned char * newBuffer = new unsigned char[outWidth*outHeight*4];

	uint32_t *const inBuffer = reinterpret_cast<uint32_t *>(inputBuffer);
	uint32_t *const outBuffer = reinterpret_cast<uint32_t *>(newBuffer);

	for (int y = 0; y < inHeight; ++y)
	{
		const int inRowPos = inWidth * y;
		const int outRowPos = outWidth * N * y;

		for (int x = 0; x < inWidth; ++x)
		{
			std::fill_n(&outBuffer[outRowPos + N * x], N, inBuffer[inRowPos + x]);
		}

		for (int c = 1; c < N; ++c)
		{
			std::copy_n(&outBuffer[outRowPos], outWidth, &outBuffer[outRowPos + outWidth * c]);
		}
	}

	delete[] inputBuffer;
	return newBuffer;
}

#ifdef HAVE_MMX
static unsigned char *hqNxAsmHelper( void (*hqNxFunction) ( int*, unsigned char*, int, int, int ),
							  const int N,
							  unsigned char *inputBuffer,
							  const int inWidth,
							  const int inHeight,
							  int &outWidth,
							  int &outHeight )
{
	outWidth = N * inWidth;
	outHeight = N *inHeight;

	static int initdone = false;

	if (!initdone)
	{
		HQnX_asm::InitLUTs();
		initdone = true;
	}

	auto pImageIn = std::make_unique<HQnX_asm::CImage>();
	auto& cImageIn = *pImageIn;
	cImageIn.SetImage(inputBuffer, inWidth, inHeight, 32);
	cImageIn.Convert32To17();

	unsigned char * newBuffer = new unsigned char[outWidth*outHeight*4];
	hqNxFunction( reinterpret_cast<int*>(cImageIn.m_pBitmap), newBuffer, cImageIn.m_Xres, cImageIn.m_Yres, outWidth*4 );
	delete[] inputBuffer;
	return newBuffer;
}
#endif

static unsigned char *hqNxHelper( void (HQX_CALLCONV *hqNxFunction) ( unsigned*, unsigned*, int, int ),
							  const int N,
							  unsigned char *inputBuffer,
							  const int inWidth,
							  const int inHeight,
							  int &outWidth,
							  int &outHeight )
{
	static int initdone = false;

	if (!initdone)
	{
		hqxInit();
		initdone = true;
	}
	outWidth = N * inWidth;
	outHeight = N *inHeight;

	unsigned char * newBuffer = new unsigned char[outWidth*outHeight*4];
	hqNxFunction( reinterpret_cast<unsigned*>(inputBuffer), reinterpret_cast<unsigned*>(newBuffer), inWidth, inHeight );
	delete[] inputBuffer;
	return newBuffer;
}


template <typename ConfigType>
void xbrzSetupConfig(ConfigType& cfg);

template <>
void xbrzSetupConfig(xbrz::ScalerCfg& cfg)
{
	cfg.luminanceWeight = xbrz_luminanceweight;
	cfg.equalColorTolerance = xbrz_equalcolortolerance;
	cfg.centerDirectionBias = xbrz_centerdirectionbias;
	cfg.dominantDirectionThreshold = xbrz_dominantdirectionthreshold;
	cfg.steepDirectionThreshold = xbrz_steepdirectionthreshold;
}

template <>
void xbrzSetupConfig(xbrz_old::ScalerCfg& cfg)
{
	cfg.luminanceWeight_ = xbrz_luminanceweight;
	cfg.equalColorTolerance_ = xbrz_equalcolortolerance;
	cfg.dominantDirectionThreshold = xbrz_dominantdirectionthreshold;
	cfg.steepDirectionThreshold = xbrz_steepdirectionthreshold;
}

template <typename ConfigType>
static unsigned char *xbrzHelper( void (*xbrzFunction) ( size_t, const uint32_t*, uint32_t*, int, int, xbrz::ColorFormat, const ConfigType&, int, int ),
							  const int N,
							  unsigned char *inputBuffer,
							  const int inWidth,
							  const int inHeight,
							  int &outWidth,
							  int &outHeight )
{
	outWidth = N * inWidth;
	outHeight = N *inHeight;

	unsigned char * newBuffer = new unsigned char[outWidth*outHeight*4];

	const int thresholdWidth  = gl_texture_hqresize_mt_width;
	const int thresholdHeight = gl_texture_hqresize_mt_height;

	ConfigType cfg;
	xbrzSetupConfig(cfg);

	const xbrz::ColorFormat colorFormat = xbrz_colorformat == 0
		? xbrz::ColorFormat::ARGB
		: xbrz::ColorFormat::ARGB_UNBUFFERED;

	if (gl_texture_hqresize_multithread
		&& inWidth  > thresholdWidth
		&& inHeight > thresholdHeight)
	{
		parallel_for(inHeight, thresholdHeight, [=, &cfg](int sliceY)
		{
			xbrzFunction(N, reinterpret_cast<uint32_t*>(inputBuffer), reinterpret_cast<uint32_t*>(newBuffer),
				inWidth, inHeight, colorFormat, cfg, sliceY, sliceY + thresholdHeight);
		});
	}
	else
	{
		xbrzFunction(N, reinterpret_cast<uint32_t*>(inputBuffer), reinterpret_cast<uint32_t*>(newBuffer),
			inWidth, inHeight, colorFormat, cfg, 0, std::numeric_limits<int>::max());
	}

	delete[] inputBuffer;
	return newBuffer;
}

// Helper: Convert float32 to IEEE 754 float16 (not handling NaN/Inf for brevity)
// Later for fp16 support
static inline uint16_t float32_to_float16(float value) 
{
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	uint32_t sign = (bits >> 16) & 0x8000;
	uint32_t mantissa = bits & 0x007FFFFF;
	int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
	if (exp <= 0) return sign; // underflow to zero
	if (exp >= 31) return sign | 0x7C00; // overflow to Inf
	return sign | (exp << 10) | (mantissa >> 13);
}

// Convert IEEE 754 float16 to float32
// Later for fp16 support
static inline float float16_to_float32(uint16_t h)
{
	uint16_t h_exp = (h & 0x7C00) >> 10;
	uint16_t h_sig = h & 0x03FF;
	uint32_t f_sgn = ((uint32_t)h & 0x8000) << 16;
	uint32_t f_exp, f_sig;

	if (h_exp == 0) 
	{
		// Zero / subnormal
		if (h_sig == 0) {
			f_exp = 0;
			f_sig = 0;
		} 
		else 
		{
			// Normalize subnormal
			h_exp = 1;
			while ((h_sig & 0x0400) == 0) 
			{
				h_sig <<= 1;
				h_exp--;
			}
			h_sig &= 0x03FF;
			f_exp = (127 - 15 - h_exp) << 23;
			f_sig = (uint32_t)h_sig << 13;
		}
	} 
	else if (h_exp == 0x1F)
	{
		// Inf/NaN
		f_exp = 0xFF << 23;
		f_sig = (uint32_t)h_sig << 13;
	} 
	else
	{
		// Normalized
		f_exp = ((h_exp - 15 + 127) & 0xFF) << 23;
		f_sig = (uint32_t)h_sig << 13;
	}
	uint32_t f = f_sgn | f_exp | f_sig;
	float result;
	std::memcpy(&result, &f, sizeof(result));
	return result;
}

// Custom logging callback for ONNX Runtime
static void ORT_API_CALL OnnxPrintfLogger(
	void* /*param*/,
	OrtLoggingLevel severity,
	const char* category,
	const char* logid,
	const char* code_location,
	const char* message)
{
	Printf("[ONNX][%s][%s][%s] %s\n", logid, category, code_location, message);
}

static bool OnnxDebug = false;
static unsigned char* OnnxHelper(int& N,
	unsigned char* inputBuffer,
	const int inWidth,
	const int inHeight,
	int& outWidth,
	int& outHeight,
	const int lump,
	bool isAlpha)

{
	static const Ort::Env env(OnnxDebug ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_ERROR, "onnx", OnnxPrintfLogger, nullptr);
	Ort::SessionOptions session_options;
	auto& api = Ort::GetApi();
	OrtCUDAProviderOptionsV2* cuda_options = nullptr;

	try
	{
		auto status = api.CreateCUDAProviderOptions(&cuda_options);
		if (status != nullptr)
		{
			Printf("ONNX Failed to create CUDA provider options: %s\n", api.GetErrorMessage(status));
			api.ReleaseStatus(status);
		} 
		else
		{
			if (OnnxDebug) Printf("ONNX Created CUDA provider options, 4GB VRAM Limit\n");
			std::vector<const char*> keys{ "device_id", "gpu_mem_limit", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace", "cudnn_conv1d_pad_to_nc1d" };
			std::vector<const char*> values{ "0", "4294967296", "kSameAsRequested", "DEFAULT", "1", "1", "1" };

			auto cudaStatus = api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());
			if (cudaStatus != nullptr)
			{
				Printf("ONNX Failed to UpdateCUDAProviderOptions: %s\n", api.GetErrorMessage(cudaStatus));
				api.ReleaseStatus(cudaStatus);
			} 
			else
			{
				if (OnnxDebug) Printf("ONNX Updated CUDA provider options\n");
				auto providerStatus = api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options);
				if (providerStatus != nullptr)
				{
					Printf("ONNX Failed to SessionOptionsAppendExecutionProvider_CUDA_V2 %s\n", api.GetErrorMessage(providerStatus));
					api.ReleaseStatus(providerStatus);
				} 
				else
				{
					if (OnnxDebug) Printf("ONNX Updated SessionOptionsAppendExecutionProvider_CUDA_V2\n");
				}
			}
		}
	} 
	catch (const Ort::Exception& ex)
	{
		Printf("ONNX: CUDA provider not available, falling back to CPU: %s\n", ex.what());
	}

	static bool model_loaded = false;
	static std::unique_ptr<Ort::Session> session;

	if (!model_loaded)
	{
		try
		{
			session = std::make_unique<Ort::Session>(env, L"model.onnx", session_options);
			if (OnnxDebug) Printf("ONNX model loaded successfully.\n");
			model_loaded = true;
		} 
		catch (const Ort::Exception& ex)
		{
			Printf("Failed to load ONNX model: %s\n", ex.what());
			model_loaded = false;
		}
	}

	// If model failed to load, skip further ONNX processing
	if (!model_loaded)
		return inputBuffer;

	// 1. Get input/output names
	static const Ort::AllocatorWithDefaultOptions allocator;
	static const auto input_name = session->GetInputNameAllocated(0, allocator);
	static const auto output_name = session->GetOutputNameAllocated(0, allocator);

	if (OnnxDebug) {
		Printf("Input name: %s\n", input_name.get());
		Printf("Output name: %s\n", output_name.get());
	}

	// 2. Prepare input tensor (convert to float32, 3 channel size)
	std::vector<int64_t> input_shape = { 3, 3, inHeight, inWidth };
	size_t input_tensor_size = 3 * 3 * inHeight * inWidth;
	std::vector<float> float32_buffer(input_tensor_size, 0.0f);

	// Convert RGBA NHWC to RGB NCHW and normalize to [0, 1]
	// Fill each batch with a single channel (R, G, B) and handle alpha==0 with nearest neighbor search
	static const int rgba_channel_map[3] = { 0, 1, 2 }; // R, G, B offsets in RGBA
	static const int rgba_alpha_index = 3;
	for (int d = 0; d < 3; ++d)
	{
		int b = rgba_channel_map[d];
		for (int h = 0; h < inHeight; ++h)
		{
			for (int w = 0; w < inWidth; ++w)
			{
				size_t nhwc_index = (h * inWidth + w) * 4;
				int alpha = inputBuffer[nhwc_index + 3];
				float value = 0.0f;

				if (alpha == 0)
				{
					// Try to find a neighbor with alpha > 0
					bool found = false;
					static const int dx[8] = { 0, 0, -1, 1, -1, 1, -1, 1 };
					static const int dy[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
					for (int d = 0; d < 8 && !found; ++d)
					{
						int nx = w + dx[d];
						int ny = h + dy[d];
						if (nx >= 0 && nx < inWidth && ny >= 0 && ny < inHeight)
						{
							size_t nidx = (ny * inWidth + nx) * 4;
							if (inputBuffer[nidx + 3] > 0)
							{
								value = static_cast<float>(inputBuffer[nidx + b]) / 255.0f;
								found = true;
							}
						}
					}
					if (!found)
					{
						value = 1.0f; // fallback: white
					}
				} else
				{
					value = static_cast<float>(inputBuffer[nhwc_index + b]) / 255.0f;
				}

				for (int c = 0; c < 3; ++c)
				{
					size_t idx = b * 3 * inHeight * inWidth + rgba_channel_map[c] * inHeight * inWidth + h * inWidth + w;
					float32_buffer[idx] = value;
				}
			}
		}
	}

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor(
		memory_info, float32_buffer.data(), float32_buffer.size() * sizeof(float),
		input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

	// 3. Run inference
	std::vector<const char*> input_names = { input_name.get() };
	std::vector<const char*> output_names = { output_name.get() };
	auto output_tensors = session->Run(
		Ort::RunOptions{ nullptr },
		input_names.data(), &input_tensor, 1,
		output_names.data(), 1);

	// 4. Get output tensor info
	Ort::Value& output_tensor = output_tensors.front();
	auto output_type_info = output_tensor.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> output_shape = output_type_info.GetShape();

	api.ReleaseCUDAProviderOptions(cuda_options);

	// 5. Set outWidth and outHeight from output shape
	// Expecting output shape: [3, 3, outH, outW]
	if (output_shape.size() == 4 && output_shape[0] == 3 && output_shape[1] == 3)
	{
		int outH = static_cast<int>(output_shape[2]);
		int outW = static_cast<int>(output_shape[3]);
		int Nw = outW / inWidth;
		int Nh = outH / inHeight;
		N = std::max({ Nw, Nh });
		if (OnnxDebug) Printf("Detected ONNX scaling factor N = %d\n", N);

		outWidth = outW;
		outHeight = outH;
		size_t pixel_count = outW * outH;

		unsigned char* newBuffer = new unsigned char[pixel_count * 4];
		if (!newBuffer)
		{
			Printf("Failed to allocate memory for ONNX output buffer.\n");
			outWidth = inWidth;
			outHeight = inHeight;
			return inputBuffer;
		}

		const float* output_data = output_tensor.GetTensorData<float>();
		for (int h = 0; h < outH; ++h)
		{
			for (int w = 0; w < outW; ++w)
			{
				size_t dst_idx = (h * outW + w) * 4;

				// For each batch, extract the upscaled channel
				unsigned char r = static_cast<unsigned char>(std::min(std::max(output_data[0 * 3 * outH * outW + 0 * outH * outW + h * outW + w], 0.0f), 1.0f) * 255.0f);
				unsigned char g = static_cast<unsigned char>(std::min(std::max(output_data[1 * 3 * outH * outW + 1 * outH * outW + h * outW + w], 0.0f), 1.0f) * 255.0f);
				unsigned char b = static_cast<unsigned char>(std::min(std::max(output_data[2 * 3 * outH * outW + 2 * outH * outW + h * outW + w], 0.0f), 1.0f) * 255.0f);

				// Alpha: nearest neighbor from input
				int src_h = h / N;
				int src_w = w / N;
				src_h = std::min(std::max(src_h, 0), inHeight - 1);
				src_w = std::min(std::max(src_w, 0), inWidth - 1);
				size_t src_alpha_idx = (src_h * inWidth + src_w) * 4 + 3;

				unsigned char a = !isAlpha ? inputBuffer[src_alpha_idx] : r;

				newBuffer[dst_idx + 0] = r;
				newBuffer[dst_idx + 1] = g;
				newBuffer[dst_idx + 2] = b;
				newBuffer[dst_idx + 3] = a;
			}
		}

		delete[] inputBuffer;
		return newBuffer;
	} 
	else
	{
		if (OnnxDebug)
		{
			Printf("ONNX output shape is unexpected: ");
			for (auto v : output_shape) Printf("%lld ", v);
			Printf("\n");
		}
		return inputBuffer;
	}
}

static unsigned char* AiScale(const int N,
	unsigned char* inputBuffer,
	const int inWidth,
	const int inHeight,
	int& outWidth,
	int& outHeight,
	const int lump)
{
	int scale = N;

	// Copy inputBuffer to alpha buffer
	const size_t inputSize = inWidth * inHeight * 4;
	auto inputBufferAlpha = new unsigned char[inputSize];
	std::memcpy(inputBufferAlpha, inputBuffer, inputSize);

	// Upscale color buffer
	inputBuffer = OnnxHelper(scale, inputBuffer, inWidth, inHeight, outWidth, outHeight, lump, false);

	// Upscale the alpha channel separately for better edge quality
	// From tests, hqNX MMX is better
	const int alphaScaleOption = 2;
	switch (alphaScaleOption)
	{
	case 0: // ScaleNX
		inputBufferAlpha = scaleNxHelper(scale == 4 ? &scale4x : scale == 3 ? &scale3x : &scale2x, scale, inputBufferAlpha, inWidth, inHeight, outWidth, outHeight);
		break;
	case 1: // ONNX
		for (int i = 0; i < inWidth * inHeight; ++i)
		{
			unsigned char a = inputBufferAlpha[i * 4 + 3];
			inputBufferAlpha[i * 4 + 0] = a;
			inputBufferAlpha[i * 4 + 1] = a;
			inputBufferAlpha[i * 4 + 2] = a;
			inputBufferAlpha[i * 4 + 3] = 255;
		}
		inputBufferAlpha = OnnxHelper(scale, inputBufferAlpha, inWidth, inHeight, outWidth, outHeight, lump, true);
		break;
	case 2:
	default:// hqNX
#ifdef HAVE_MMX
		auto func = &HQnX_asm::hq2x_32;
		if (scale == 4)
			func = &HQnX_asm::hq4x_32;
		else if (scale == 3)
			func = &HQnX_asm::hq3x_32;
		inputBufferAlpha = hqNxAsmHelper(func, scale, inputBufferAlpha, inWidth, inHeight, outWidth, outHeight);
#else
		inputBufferAlpha = hqNxHelper(scale == 4 ? &hq4x_32 : scale == 3 ? &hq3x_32 : &hq2x_32, scale, inputBufferAlpha, inWidth, inHeight, outWidth, outHeight);
#endif //HAVE_MMX
	}

	// Combine upscaled RGB and alpha
	auto outputBuffer = new unsigned char[outWidth * outHeight * 4];
	for (int i = 0; i < outWidth * outHeight; ++i)
	{
		outputBuffer[i * 4 + 0] = inputBuffer[i * 4 + 0];
		outputBuffer[i * 4 + 1] = inputBuffer[i * 4 + 1];
		outputBuffer[i * 4 + 2] = inputBuffer[i * 4 + 2];
		outputBuffer[i * 4 + 3] = inputBufferAlpha[i * 4 + 3];
	}

	delete[] inputBufferAlpha;
	delete[] inputBuffer;
	return outputBuffer;
}

static void xbrzOldScale(size_t factor, const uint32_t* src, uint32_t* trg, int srcWidth, int srcHeight, xbrz::ColorFormat colFmt, const xbrz_old::ScalerCfg& cfg, int yFirst, int yLast)
{
	xbrz_old::scale(factor, src, trg, srcWidth, srcHeight, cfg, yFirst, yLast);
}


//===========================================================================
// 
// [BB] Upsamples the texture in texbuffer.mBuffer, frees texbuffer.mBuffer and returns
//  the upsampled buffer.
//
//===========================================================================

void FTexture::CreateUpsampledTextureBuffer(FTextureBuffer &texbuffer, bool hasAlpha, bool checkonly)
{
	// [BB] Make sure that inWidth and inHeight denote the size of
	// the returned buffer even if we don't upsample the input buffer.

	int inWidth = texbuffer.mWidth;
	int inHeight = texbuffer.mHeight;

	int type = gl_texture_hqresizemode;
	int mult = gl_texture_hqresizemult;
#ifdef HAVE_MMX
	// hqNx MMX does not preserve the alpha channel so fall back to C-version for such textures
	if (hasAlpha && type == 3)
	{
		type = 2;
	}
#endif
	// These checks are to ensure consistency of the content ID.
	if (mult < 2 || mult > 6 || type < 1 || type > 6) return;
	if (type < 4 && mult > 4) mult = 4;

	if (!checkonly)
	{
		if (type == 1)
		{
			if (mult == 2)
				texbuffer.mBuffer = scaleNxHelper(&scale2x, 2, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else if (mult == 3)
				texbuffer.mBuffer = scaleNxHelper(&scale3x, 3, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else if (mult == 4)
				texbuffer.mBuffer = scaleNxHelper(&scale4x, 4, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else return;
		}
		else if (type == 2)
		{
			if (mult == 2)
				texbuffer.mBuffer = hqNxHelper(&hq2x_32, 2, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else if (mult == 3)
				texbuffer.mBuffer = hqNxHelper(&hq3x_32, 3, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else if (mult == 4)
				texbuffer.mBuffer = hqNxHelper(&hq4x_32, 4, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else return;
		}
#ifdef HAVE_MMX
		else if (type == 3)
		{
			if (mult == 2)
				texbuffer.mBuffer = hqNxAsmHelper(&HQnX_asm::hq2x_32, 2, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else if (mult == 3)
				texbuffer.mBuffer = hqNxAsmHelper(&HQnX_asm::hq3x_32, 3, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else if (mult == 4)
				texbuffer.mBuffer = hqNxAsmHelper(&HQnX_asm::hq4x_32, 4, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			else return;
		}
#endif
		else if (type == 4)
			texbuffer.mBuffer = xbrzHelper(xbrz::scale, mult, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
		else if (type == 5)
			texbuffer.mBuffer = xbrzHelper(xbrzOldScale, mult, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
		else if (type == 6) {
			mult = 2;
			texbuffer.mBuffer = AiScale(mult, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight, this->GetSourceLump());
			//texbuffer.mBuffer = normalNx(mult, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
		}
		else
			return;
	}
	else
	{
		texbuffer.mWidth *= mult;
		texbuffer.mHeight *= mult;
	}
	// Encode the scaling method in the content ID.
	FContentIdBuilder contentId;
	contentId.id = texbuffer.mContentId;
	contentId.scaler = type;
	contentId.scalefactor = mult;
	texbuffer.mContentId = contentId.id;
}

//===========================================================================
// 
// This was pulled out of the above function to allow running these
// checks before the texture is passed to the render state.
//
//===========================================================================

void calcShouldUpscale(FGameTexture *tex)
{
	tex->SetUpscaleFlag(0);
	// [BB] Don't resample if width * height of the input texture is bigger than gl_texture_hqresize_maxinputsize squared.
	const int maxInputSize = gl_texture_hqresize_maxinputsize;
	if (tex->GetTexelWidth() * tex->GetTexelHeight() > maxInputSize * maxInputSize)
		return;

	// [BB] Don't try to upsample textures based off FCanvasTexture. (This should never get here in the first place!)
	if (tex->isHardwareCanvas())
		return;

	// already scaled?
	if (tex->GetScaleX() >= 2.f || tex->GetScaleY() > 2.f)
		return;

	tex->SetUpscaleFlag(1);
}