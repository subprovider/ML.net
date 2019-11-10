using System;
using System.IO;
using Microsoft.ML;

namespace TaxiFarePrediction
{
    class Program
    {
        #region Machine Running Train, Test Data 
        // En
        private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        // Ko
        //private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "trainDataKo.csv");
        //private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "testDataKo.tsv");
        
        private static string _modelPath => Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");
        #endregion


        public static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _trainDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);

        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            // FareAmount 열은 예측할(모델의 출력) Label입니다. CopyColumnsEstimator 변환 클래스를 사용하여 FareAmount를 복사
            // 모델을 학습시키는 알고리즘에는 숫자 기능이 필요하므로 범주 데이터(VendorId, RateCode 및 PaymentType) 값을 
            // 숫자(VendorIdEncoded, RateCodeEncoded 및 PaymentTypeEncoded)로 변환해야 합니다. 
            // 이 작업을 수행하려면 각 열의 값마다 다른 숫자 키 값을 할당하는 OneHotEncodingTransformer 변환 클래스를 사용
            // 데이터 준비의 마지막 단계에서는 mlContext.Transforms.Concatenate 변환 클래스를 사용하여 모든 기능 열을 Features 열에 결합
            // Train()에서 다음 코드 줄에 다음을 추가하여 FastTreeRegressionTrainer 기계 학습 작업
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                            .Append(mlContext.Regression.Trainers.FastTree());

            // Fit() 메서드는 데이터 세트를 변환하고 학습을 적용하여 모델을 학습합니다.
            var model = pipeline.Fit(dataView);

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // 테스트 데이터 세트를 로드합니다.
            // 회귀 평가자를 만듭니다.
            // 모델을 평가하고 메트릭을 만듭니다.
            // 메트릭을 표시합니다.
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            // Transform() 메서드는 테스트 데이터 세트 입력 행에 대한 예측을 수행합니다.
            var predictions = model.Transform(dataView);

            // RegressionContext.Evaluate 메서드는 지정된 데이터 세트를 사용하여 PredictionModel에 대한 품질 메트릭을 계산합니다. 
            // 회귀 평가자가 계산한 전체 메트릭이 포함된 RegressionMetrics 개체를 반환합니다.
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            // RSquared는 회귀 모델의 다른 평가 메트릭입니다. RSquared에서는 0과 1 사이의 값을 사용합니다. 해당 값이 1에 가까울수록 더 나은 모델입니다.
            // RMS는 회귀 모델의 평가 메트릭 중 하나입니다. RMS가 낮을수록 더 나은 모델입니다

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            // 테스트 데이터의 단일 댓글을 만듭니다.
            // 테스트 데이터를 기준으로 요금 금액을 예측합니다.
            // 보고를 위해 테스트 데이터 및 예측을 결합합니다.
            // 예측 결과를 표시합니다.

            // PredictionEngine은 데이터의 단일 인스턴스에 대한 예측을 수행할 수 있는 편리한 API입니다. 
            // PredictionEngine은 스레드로부터 안전하지 않습니다. 
            // 단일 스레드 또는 프로토타입 환경에서 사용할 수 있습니다. 
            // 프로덕션 환경에서 성능 및 스레드 보안을 개선하려면 PredictionEnginePool 서비스를 사용합니다. 
            // 이 서비스는 애플리케이션 전체에서 사용할 PredictionEngine 개체의 ObjectPool을 만듭니다. 
            // ASP.NET Core Web API에서 PredictionEnginePool을 사용하는 방법에 대한 이 가이드를 참조하세요.
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            // Predict() 함수는 단일 데이터 인스턴스에 대한 예측을 수행합니다.
            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");

        }
    }
}
