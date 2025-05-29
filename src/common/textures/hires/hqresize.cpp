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

#include <array>
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

CUSTOM_CVAR(Float, gl_texture_hqresize_aiscale_sharpen, 0.06f, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL)
{
	TexMan.FlushAll();
	UpdateUpscaleMask();
}
CUSTOM_CVAR(Int, gl_texture_hqresize_aiscale_alpha_algorithm, 2, CVAR_ARCHIVE | CVAR_GLOBALCONFIG | CVAR_NOINITCALL)
{
	TexMan.FlushAll();
	UpdateUpscaleMask();
}
CVAR(Bool, gl_texture_hqresize_aiscale_use_gpu, true, CVAR_ARCHIVE | CVAR_GLOBALCONFIG);
CVAR(Bool, gl_texture_hqresize_aiscale_debug, false, CVAR_ARCHIVE | CVAR_GLOBALCONFIG);
CVAR(Float, gl_texture_hqresize_aiscale_vram_limit_gb, 3.0f, CVAR_ARCHIVE | CVAR_GLOBALCONFIG);

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

namespace {
	struct OrtCudaOptionsDeleter {
		void operator()(OrtCUDAProviderOptionsV2* ptr) const noexcept {
			if (ptr) {
				auto& api = Ort::GetApi();
				api.ReleaseCUDAProviderOptions(ptr);
			}
		}
	};
}

static const bool OnnxDebug = false;
static unsigned char* OnnxHelper(int& N,
	unsigned char* inputBuffer,
	const int inWidth,
	const int inHeight,
	int& outWidth,
	int& outHeight,
	bool isAlpha,
	bool isTiling)
{
	static const Ort::Env env(OnnxDebug ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_ERROR, "onnx", OnnxPrintfLogger, nullptr);
	// Static session options and CUDA options, initialized once
	static std::unique_ptr<OrtCUDAProviderOptionsV2, OrtCudaOptionsDeleter> cuda_options;
	static Ort::SessionOptions session_options;

	static bool cuda_initialized = false;
	static bool provider_selected = false;

	if (!provider_selected)
	{
		auto& api = Ort::GetApi();
		// Try GPU first
		if (gl_texture_hqresize_aiscale_use_gpu)
		{
			// Try CUDA first
			try
			{
				if (!cuda_initialized)
				{
					OrtCUDAProviderOptionsV2* raw_cuda_options = nullptr;
					auto status = api.CreateCUDAProviderOptions(&raw_cuda_options);
					if (status == nullptr)
					{
						// Set CUDA options
						std::string vram_limit_str = std::to_string(static_cast<long long>(static_cast<float>(gl_texture_hqresize_aiscale_vram_limit_gb) * 1024 * 1024 * 1024));
						std::array<const char*, 7> keys = { "device_id", "gpu_mem_limit", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace", "cudnn_conv1d_pad_to_nc1d" };
						std::array<const char*, 7> values = { "0", vram_limit_str.c_str(), "kSameAsRequested", "DEFAULT", "1", "1", "1"};
						auto cudaStatus = api.UpdateCUDAProviderOptions(raw_cuda_options, keys.data(), values.data(), keys.size());
						if (cudaStatus == nullptr)
						{
							cuda_options.reset(raw_cuda_options);
							auto providerStatus = api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, raw_cuda_options);
							if (providerStatus == nullptr)
							{
								Printf("ONNX: Using CUDA provider, %.1fGB VRAM limit\n", static_cast<float>(gl_texture_hqresize_aiscale_vram_limit_gb));
								provider_selected = true;
							}
							else
							{
								Printf("ONNX: CUDA is unavailable, failed to set session options\n");
								api.ReleaseStatus(providerStatus);
							}
						} 
						else
						{
							Printf("ONNX: CUDA is unavailable, failed to update CUDA provider options\n");
							api.ReleaseStatus(cudaStatus);
						}
					} 
					else
					{
						Printf("ONNX: CUDA is unavailable, failed to create CUDA provider options\n");
						api.ReleaseStatus(status);
					}
				}
				cuda_initialized = true;
			}
			catch (const Ort::Exception& ex) {
				Printf("ONNX: CUDA provider not available: %s\n", ex.what());
				cuda_initialized = true;
			}
		}
		
		// If CUDA is unavailable, CPU will be used by default (no need to add explicitly)
		if (!provider_selected)
		{
			Printf("ONNX: Using default CPU provider\n");
			provider_selected = true;
		}
	}

	static bool model_loaded = false;
	static bool model_initialized = false;
	static std::unique_ptr<Ort::Session> session;

	if (!model_initialized)
	{
		try
		{
			session = std::make_unique<Ort::Session>(env, L"model.onnx", session_options);
			Printf("ONNX model loaded successfully.\n");
			model_loaded = true;
		} 
		catch (const Ort::Exception& ex)
		{
			Printf("Failed to load ONNX model, no upscaling available: %s\n", ex.what());
			model_loaded = false;
		}
		model_initialized = true;
	}

	// If model failed to load, skip further ONNX processing
	if (!model_loaded) {
		N = 1;
		return inputBuffer;
	}

	// Get input/output names
	static const Ort::AllocatorWithDefaultOptions allocator;
	static const auto input_name = session->GetInputNameAllocated(0, allocator);
	static const auto output_name = session->GetOutputNameAllocated(0, allocator);

	// Tiling logic
	const int pad = 1;
	const int paddedWidth = inWidth + 2 * pad;
	const int paddedHeight = inHeight + 2 * pad;
	std::vector<unsigned char> paddedInput(paddedWidth * paddedHeight * 4);

	if (isTiling) {
		// Fill paddedInput with wrapped pixels
		for (int y = 0; y < paddedHeight; ++y) {
			const int srcY = (y - pad + inHeight) % inHeight;
			for (int x = 0; x < paddedWidth; ++x) {
				const int srcX = (x - pad + inWidth) % inWidth;
				for (int c = 0; c < 4; ++c) {
					paddedInput[(y * paddedWidth + x) * 4 + c] =
						inputBuffer[(srcY * inWidth + srcX) * 4 + c];
				}
			}
		}
	}

	const unsigned char* modelInput = isTiling ? paddedInput.data() : inputBuffer;
	const int modelInWidth = paddedWidth;
	const int modelInHeight = paddedHeight;

	// Prepare input tensor (3 batches, 3 channels), float32, input shape [3, 3, inW, inH]
	const std::vector<int64_t> input_shape = { 3, 3, modelInHeight, modelInWidth };
	const size_t input_tensor_size = 3 * 3 * modelInHeight * modelInWidth;
	std::vector<float> float32_buffer(input_tensor_size, 0.0f);

	// Convert RGBA NHWC to RGB NCHW and normalize to [0, 1]
	// Fill each batch with a single channel (R, G, B) and handle alpha==0 with nearest neighbor search
	static const std::array<int, 3> rgba_channel_map = { 0, 1, 2 }; // R, G, B offsets in RGBA
	static const int rgba_alpha_index = 3;
	for (int d = 0; d < 3; ++d)
	{
		const int b = rgba_channel_map[d];
		for (int h = 0; h < modelInHeight; ++h)
		{
			for (int w = 0; w < modelInWidth; ++w)
			{
				const size_t nhwc_index = (h * modelInWidth + w) * 4;
				const int alpha = modelInput[nhwc_index + rgba_alpha_index];
				float value = 0.0f;

				if (alpha == 0)
				{
					// Nearest neighbor search in padded input
					bool found = false;
					static const std::array<int, 8> dx = { 0, 0, -1, 1, -1, 1, -1, 1 };
					static const std::array<int, 8> dy = { -1, 1, 0, 0, -1, -1, 1, 1 };
					for (int dv = 0; dv < 8 && !found; ++dv)
					{
						const int nx = w + dx[dv];
						const int ny = h + dy[dv];
						if (nx >= 0 && nx < modelInWidth && ny >= 0 && ny < modelInHeight)
						{
							const size_t nidx = (ny * modelInWidth + nx) * 4;
							if (modelInput[nidx + 3] > 0)
							{
								value = static_cast<float>(modelInput[nidx + b]) / 255.0f;
								found = true;
							}
						}
					}
					if (!found)
					{
						value = 0.0f; // fallback: black
					}
				}
				else
				{
					value = static_cast<float>(modelInput[nhwc_index + b]) / 255.0f;
				}

				for (int c = 0; c < 3; ++c)
				{
					const size_t idx = b * 3 * modelInHeight * modelInWidth + rgba_channel_map[c] * modelInHeight * modelInWidth + h * modelInWidth + w;
					float32_buffer[idx] = value;
				}
			}
		}
	}

	static const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	const Ort::Value input_tensor = Ort::Value::CreateTensor(
		memory_info, float32_buffer.data(), float32_buffer.size() * sizeof(float),
		input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

	// Run inference
	static const std::vector<const char*> input_names = { input_name.get() };
	static const std::vector<const char*> output_names = { output_name.get() };
	const auto output_tensors = session->Run(
		Ort::RunOptions{ nullptr },
		input_names.data(), &input_tensor, 1,
		output_names.data(), 1);

	// Get output tensor info
	const Ort::Value& output_tensor = output_tensors.front();
	const auto output_type_info = output_tensor.GetTensorTypeAndShapeInfo();
	const std::vector<int64_t> output_shape = output_type_info.GetShape();

	// Process output
	// Expecting output shape: [3, 3, outH, outW]
	if (output_shape.size() == 4 && output_shape[0] == 3 && output_shape[1] == 3)
	{
		const int outH = static_cast<int>(output_shape[2]);
		const int outW = static_cast<int>(output_shape[3]);
		const int Nw = outW / modelInWidth;
		const int Nh = outH / modelInHeight;
		N = std::max({ Nw, Nh });

		// Calculate crop region
		const int cropX = pad * N;
		const int cropY = pad * N;
		const int cropW = inWidth * N;
		const int cropH = inHeight * N;

		outWidth = cropW;
		outHeight = cropH;
		const size_t pixel_count = cropW * cropH;

		auto newBuffer = new unsigned char[pixel_count * 4];
		const float* output_data = output_tensor.GetTensorData<float>();

		for (int h = 0; h < cropH; ++h)
		{
			for (int w = 0; w < cropW; ++w)
			{
				const int oh = h + cropY;
				const int ow = w + cropX;
				const size_t dst_idx = (h * cropW + w) * 4;

				const unsigned char r = static_cast<unsigned char>(std::min(std::max(output_data[0 * 3 * outH * outW + 0 * outH * outW + oh * outW + ow], 0.0f), 1.0f) * 255.0f);
				const unsigned char g = static_cast<unsigned char>(std::min(std::max(output_data[1 * 3 * outH * outW + 1 * outH * outW + oh * outW + ow], 0.0f), 1.0f) * 255.0f);
				const unsigned char b = static_cast<unsigned char>(std::min(std::max(output_data[2 * 3 * outH * outW + 2 * outH * outW + oh * outW + ow], 0.0f), 1.0f) * 255.0f);

				// Alpha: nearest neighbor from input (original, not padded)
				const int src_h = std::min(std::max(h / N, 0), inHeight - 1);
				const int src_w = std::min(std::max(w / N, 0), inWidth - 1);
				const size_t src_alpha_idx = (src_h * inWidth + src_w) * 4 + 3;

				const unsigned char a = !isAlpha ? inputBuffer[src_alpha_idx] : r;

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
		if (gl_texture_hqresize_aiscale_debug)
		{
			Printf("ONNX output shape is unexpected: in ");
			for (const auto& v : input_shape) Printf("%lld ", v);
			Printf(", out ");
			for (const auto& v : output_shape) Printf("%lld ", v);
			Printf("\n");
		}
		N = 1;
		return inputBuffer;
	}
}

static void SharpenBuffer(unsigned char* buffer, int width, int height, float strength)
{
	if (strength <= 0.0f) return;

	std::vector<unsigned char> temp(buffer, buffer + width * height * 4);

	// 3x3 sharpening kernel: center = 5, neighbors = -1
	// out = (5 * center - sum(neighbors)) * strength + center * (1-strength)
	for (int y = 1; y < height - 1; ++y)
	{
		for (int x = 1; x < width - 1; ++x)
		{
			for (int c = 0; c < 3; ++c) // Only RGB, not alpha
			{
				const int idx = (y * width + x) * 4 + c;
				int sum = 0;
				sum += temp[((y - 1) * width + (x)) * 4 + c];
				sum += temp[((y + 1) * width + (x)) * 4 + c];
				sum += temp[((y)*width + (x - 1)) * 4 + c];
				sum += temp[((y)*width + (x + 1)) * 4 + c];
				const int center = temp[idx];
				int sharpened = static_cast<int>(
					(center * 5 - sum) * strength + center * (1.0f - strength) + 0.5f
					);
				buffer[idx] = static_cast<unsigned char>(std::clamp(sharpened, 0, 255));
			}
		}
	}
}

static unsigned char* AiScale(int& N,
	unsigned char* inputBuffer,
	const int inWidth,
	const int inHeight,
	int& outWidth,
	int& outHeight)
{
	int scale = N;

	// Copy inputBuffer for alpha processing
	const size_t inputSize = inWidth * inHeight * 4;
	auto inputBufferAlpha = new unsigned char[inputSize];
	std::memcpy(inputBufferAlpha, inputBuffer, inputSize);
	const int inAlphaWidth = inWidth;
	const int inAlphaHeight = inHeight;
	int outAlphaWidth = outWidth;
	int outAlphaHeight = outHeight;

	// Upscale color buffer
	const auto inputBufferPtr = inputBuffer;
	inputBuffer = OnnxHelper(scale, inputBuffer, inWidth, inHeight, outWidth, outHeight, false, true);

	// If scaling failed (same pointer on return) - return input buffer
	if (inputBuffer == inputBufferPtr)
	{
		delete[] inputBufferAlpha;
		N = 1;
		return inputBuffer;
	}

	// Post process color buffer
	SharpenBuffer(inputBuffer, outWidth, outHeight, gl_texture_hqresize_aiscale_sharpen);

	// Upscale the alpha channel separately for better edge quality
	// From tests, hqNX MMX is better
	if (scale > 1)
	{
		const int alphaScaleOption = gl_texture_hqresize_aiscale_alpha_algorithm;
		switch (alphaScaleOption)
		{
		case 0: // ONNX
			for (int i = 0; i < inWidth * inHeight; ++i)
			{
				const unsigned char a = inputBufferAlpha[i * 4 + 3];
				inputBufferAlpha[i * 4 + 0] = a;
				inputBufferAlpha[i * 4 + 1] = a;
				inputBufferAlpha[i * 4 + 2] = a;
				inputBufferAlpha[i * 4 + 3] = 255;
			}
			inputBufferAlpha = OnnxHelper(scale, inputBufferAlpha, inAlphaWidth, inAlphaHeight, outAlphaWidth, outAlphaHeight, true, true);
			break;
		case 1: // ScaleNX
			inputBufferAlpha = scaleNxHelper(scale == 2 ? &scale2x : scale == 3 ? &scale3x : &scale4x, scale, inputBufferAlpha, inAlphaWidth, inAlphaHeight, outAlphaWidth, outAlphaHeight);
			break;
		case 2:
		default:// hqNX
#ifdef HAVE_MMX
			auto func = &HQnX_asm::hq2x_32;
			switch (scale)
			{
			case 2:
				//func = &HQnX_asm::hq2x_32;
				break;
			case 3:
				func = &HQnX_asm::hq3x_32;
				break;
			case 4:
			default:
				func = &HQnX_asm::hq4x_32;
				break;
			}
			inputBufferAlpha = hqNxAsmHelper(func, scale, inputBufferAlpha, inAlphaWidth, inAlphaHeight, outAlphaWidth, outAlphaHeight);
#else
			inputBufferAlpha = hqNxHelper(scale == 2 ? &hq2x_32 : scale == 3 ? &hq3x_32 : &hq4x_32, scale, inputBufferAlpha, inAlphaWidth, inAlphaHeight, outAlphaWidth, outAlphaHeight);
#endif //HAVE_MMX
			break;
		}
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
	N = scale;
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
		else if (type == 6)
		{
			const int oldMult = mult;
			texbuffer.mBuffer = AiScale(mult, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			//if (oldMult == 4 && mult == 2) // Upscale twice if model is only 2x, commented out since its lower quality, use proper 4x model
			//{
			//	const int inWidth = texbuffer.mWidth;
			//	const int inHeight = texbuffer.mHeight;
			//	texbuffer.mBuffer = AiScale(mult, texbuffer.mBuffer, inWidth, inHeight, texbuffer.mWidth, texbuffer.mHeight);
			//}
				
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