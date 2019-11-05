using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace GitHubIssueClassification
{
    public class Program
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

        private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues_train.csv");
        private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            Evaluate(_trainingDataView.Schema);
            PredictIssue();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
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

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

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
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

    }
}
