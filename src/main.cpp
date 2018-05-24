#include <iostream>

#include <Windows.h>
#include <winerror.h>

#include <limits>

#include <D3D12.h>
#include <winml.h>
#include <DXGI.h>

#include "cnpy.h"

#pragma comment (lib, "d3d12.lib")
#pragma comment (lib, "dxgi.lib")
#pragma comment (lib, "winml.lib")

#define HRESCHECK(x) if(x != S_OK) return x;

class WinMlModel
{
public:
	HRESULT Init(const std::wstring & pathToModel)
	{
		HRESCHECK(InitDxDevice());

		HRESCHECK(InitMl(pathToModel));

		return S_OK;
	}

	void Deinit()
	{
		DeinitMl();
	}

	HRESULT BindInput(const UINT & id, const size_t & size, void * pData)
	{
		WINML_VARIABLE_DESC * inputVarDesk;
		WINML_BINDING_DESC bindInputDescriptor = {};

		HRESCHECK(m_mlModel->EnumerateModelInputs(id, &inputVarDesk));

		bindInputDescriptor.Name = inputVarDesk->Name;
		bindInputDescriptor.BindType = WINML_BINDING_TYPE::WINML_BINDING_TENSOR;
		bindInputDescriptor.Tensor.DataType = inputVarDesk->Tensor.ElementType;
		bindInputDescriptor.Tensor.NumDimensions = inputVarDesk->Tensor.NumDimensions;
		bindInputDescriptor.Tensor.pShape = inputVarDesk->Tensor.pShape;
		bindInputDescriptor.Tensor.DataSize = size;
		bindInputDescriptor.Tensor.pData = pData;

		m_mlContext->BindValue(&bindInputDescriptor);

		return S_OK;


	}

	HRESULT BindOutput(const UINT & id,const size_t & size, void * pData)
	{
		WINML_VARIABLE_DESC * outVarDesk;
		WINML_BINDING_DESC bindOutputDescriptor = {};

		HRESCHECK(m_mlModel->EnumerateModelOutputs(id, &outVarDesk));

		bindOutputDescriptor.Name = outVarDesk->Name;
		bindOutputDescriptor.BindType = WINML_BINDING_TYPE::WINML_BINDING_TENSOR;
		bindOutputDescriptor.Tensor.DataType = outVarDesk->Tensor.ElementType;
		bindOutputDescriptor.Tensor.NumDimensions = outVarDesk->Tensor.NumDimensions;
		bindOutputDescriptor.Tensor.pShape = outVarDesk->Tensor.pShape;

		bindOutputDescriptor.Tensor.DataSize = size;
		bindOutputDescriptor.Tensor.pData = pData;

		m_mlContext->BindValue(&bindOutputDescriptor);

		return S_OK;
	}


	HRESULT Evaluate()
	{
		HRESCHECK(m_mlRuntime->EvaluateModel(m_mlContext));

		return S_OK;
	}

private:

	HRESULT InitDxDevice()
	{
		IDXGIFactory * pFactory;
		HRESCHECK(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory)));


		IDXGIAdapter * adapter;

		{
			UINT adapterIndex{ 0 };
			HRESCHECK(pFactory->EnumAdapters(adapterIndex, &adapter));

			DXGI_ADAPTER_DESC desc;
			ZeroMemory(&desc, sizeof(desc));

			HRESCHECK(adapter->GetDesc(&desc));

			wprintf(L"%s\n", desc.Description);
		}

		HRESCHECK(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)));

		return S_OK;
	}

	HRESULT InitMl(const std::wstring & pathToModel)
	{
		HRESCHECK(WinMLCreateRuntime(&m_mlRuntime));

		HRESCHECK(m_mlRuntime->CreateEvaluationContext(m_device, &m_mlContext));

		HRESCHECK(m_mlRuntime->LoadModel(pathToModel.c_str(), &m_mlModel));
	}

	void DeinitMl() {
		m_mlModel->Release();
		m_mlContext->Release();
		m_mlRuntime->Release();
	}

	ID3D12Device * m_device = nullptr;
	IWinMLRuntime * m_mlRuntime = nullptr;
	IWinMLEvaluationContext * m_mlContext = nullptr;
	IWinMLModel * m_mlModel = nullptr;
};


cnpy::npz_t LoadNpy(const std::string & path)
{
	return cnpy::npz_load(path);
}


void RunOnnxTest(const std::wstring & modelPath)
{
	WinMlModel model;

	FILE * f = fopen("Log.txt", "w");

	if (model.Init(modelPath) != S_OK)
	{
		fprintf(f, "Error: WinML init fail\n");
	}

	float dataOut[10] = { 
		std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
		, std::numeric_limits<float>::quiet_NaN()
	};


	cnpy::npz_t arr = LoadNpy("../res/mnist/test_data_0.npz");
	auto input = arr["inputs"];
	float * fdat = input.data<float>();

	fprintf(f, "Input:\n");
	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			fprintf(f, "%.0f\t", fdat[i * 28 + j]);
		}
		fprintf(f, "\n");
	}

	if (model.BindInput(0, input.num_bytes(), fdat) != S_OK)
	{
		fprintf(f, "Error: WinML bind input fail\n");
	}

	if (model.BindOutput(0, sizeof(dataOut), dataOut))
	{
		fprintf(f, "Error: WinML bind output fail\n");
	}

	if (model.Evaluate() != S_OK)
	{
		fprintf(f, "Error: WinML evaluate fail\n");
	}


	fprintf(f, "Output:\n");

	for (int i = 0; i < 10; ++i)
	{
		fprintf(f,"%d: prediction: %f\n",i, dataOut[i]);
	}


	model.Deinit();
	fclose(f);
}



int main()
{
	RunOnnxTest(L"../res/mnist/mnist.onnx");

	return 0;
}