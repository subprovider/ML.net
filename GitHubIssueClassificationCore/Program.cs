using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace GitHubIssueClassificationCore
{
    class Program
    {

        /*
         * _trainDataPath에는 모델을 학습시키는 데 사용되는 데이터 세트의 경로가 포함됩니다.
         * _testDataPath에는 모델을 평가하는 데 사용되는 데이터 세트의 경로가 포함됩니다.
         * _modelPath에는 학습된 모델이 저장되는 경로가 포함됩니다.
         * _mlContext는 처리 컨텍스트를 제공하는 MLContext입니다.
         * _trainingDataView는 학습 데이터 세트를 처리하는 데 사용되는 IDataView입니다.
         * _predEngine은 단일 예측에 사용되는 PredictionEngine<TSrc,TDst>입니다.
         */

        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        //private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        //private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        //private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        #region Machine Running Train, Test Data 
        // En
        //private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues-train.csv");
        //private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues-test.tsv");
        // Ko
        private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "trainDataKo.csv");
        private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "testDataKo.tsv");


        private static string _modelPath => Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");
        #endregion

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            // LoadFromTextFile  인자에  separatorChar 가 정의되지 않으면 default Tab 임
            //_trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
            //var pipeline = ProcessData();
            //var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            //Evaluate(_trainingDataView.Schema);
            //PredictIssue();

            PredictIssueTest();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            // "Label" 은 예측된 결과로 지정할 컬럼을 의미하며, "Title", "Description" 예측을 위한 feature 로 Append 했다.
            //  MapValueToKey() 메서드를 사용하여 Area 열을 숫자 키 형식 Label 열로 변환하고(분류 알고리즘에서 허용하는 형식) 새 데이터 세트 열로 추가
            //  텍스트(Title 및 Description) 열을 TitleFeaturized 및 DescriptionFeaturized 각각에서 숫자 벡터로 변환하는 mlContext.Transforms.Text.FeaturizeText를 호출
            //  Concatenate() 메서드를 사용하여 모든 기능 열을 Features 열에 결합합니다. 기본적으로, 학습 알고리즘은 Features 열의 기능만 처리합니다
            //  AppendCacheCheckpoint를 추가하여 DataView를 캐시합니다. 따라서 해당 캐시를 사용하여 데이터를 여러 번 반복하면 다음 코드를 사용하는 것처럼 성능이 향상
            //  작거나 중간 규모의 데이터 세트에서는 AppendCacheCheckpoint를 사용하여 학습 시간을 단축하세요. 규모가 매우 큰 데이터 세트를 다룰 때는 사용하지 마세요(AppendCacheCheckpoint()를 제거)
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                            .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                            .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

            // Evaluate() 메서드는 지정된 데이터 세트를 사용하여 모델에 대한 품질 메트릭을 계산합니다. 
            // 다중 클래스 분류 평가자가 계산한 전체 메트릭을 포함하는 MulticlassClassificationMetrics 개체를 반환합니다.
            // 모델의 품질을 확인하기 위해 메트릭을 표시하려면 먼저 해당 메트릭을 가져와야 합니다.
            // 기계 학습 _trainedModel 글로벌 변수(ITransformer)의 Transform() 메서드를 사용하여 기능을 입력하고 예측을 반환

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            // MicroAccuracy : 모든 샘플 클래스 쌍은 정확도 메트릭에 동일하게 기여합니다. 마이크로 정확도를 가능한 한 1에 가깝게 합니다.
            // MacroAccuracy : 모든 클래스 정확도 메트릭에 동일하게 기여합니다. 소수 클래스는 큰 클래스와 같은 가중치를 부여받습니다. 매크로 정확도를 가능한 한 1에 가깝게 합니다.
            // LogLoss : 로그 손실을 가능한 한 0에 가깝게 합니다
            // LogLossReduction : 로그 손실 감소 - [-inf, 100]의 범위입니다. 여기서 100은 완벽한 예측이고 0은 평균 예측을 나타냅니다. 로그 손실 감소를 가능한 한 0에 가깝게 합니다.

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        private static void PredictIssue()
        {
            // 저장된 모델 로드
            // 테스트 데이터의 단일 문제를 만듭니다.
            // 테스트 데이터를 기반으로 영역을 예측합니다.
            // 보고를 위해 테스트 데이터 및 예측을 결합합니다.
            // 예측 결과를 표시합니다.

            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue()
            {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing"
            };


            // PredictionEngine은 데이터의 단일 인스턴스에 대한 예측을 수행할 수 있는 편리한 API입니다. 
            // PredictionEngine은 스레드로부터 안전하지 않습니다. 
            // 단일 스레드 또는 프로토타입 환경에서 사용할 수 있습니다. 
            // 프로덕션 환경에서 성능 및 스레드 보안을 개선하려면 PredictionEnginePool 서비스를 사용합니다. 
            // 이 서비스는 애플리케이션 전체에서 사용할 PredictionEngine 개체의 ObjectPool을 만듭니다. 
            // ASP.NET Core Web API에서 PredictionEnginePool을 사용하는 방법에 대한 이 가이드를 참조하세요.
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        private static void PredictIssueTest()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue()
            {
                Title = "제중당한약방",
                Description = "한약방"
            };

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

    }
}
