using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysisDemo
{
    class Program
    {
        //private static string trainDataPath = @"D:\csharpProject\MLnetApp\FirstMLnetProject\SentimentAnalysisDemo\data\yelp_labelled.txt";
        //private static string trainDataPath = @"D:\csharpProject\MLnetApp\FirstMLnetProject\SentimentAnalysisDemo\data\trainData.tsv";
        //private static string trainDataPath = @"D:\csharpProject\MLnetApp\FirstMLnetProject\SentimentAnalysisDemo\data\trainkorean.txt";
        //private static string testDataPath = @"D:\csharpProject\MLnetApp\FirstMLnetProject\SentimentAnalysisDemo\data\testData.tsv";

        /*
         *  train data 는 최소 50건이 되야 positivie, negativie 에 대한 학습을 시킬 수 있다.
         */
       // private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "data", "trainkorean.txt");
        private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "data", "senitimentKorean.txt");

        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "data", "testData.tsv");

        static void Main(string[] args)
        {
            runML();


        }

        private static void runML()
        {
            MLContext mlContext = new MLContext();

            TrainTestData splitDataView = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model);

            UseModelWithBatchItems(mlContext, model);

        }


        public static TrainTestData LoadData(MLContext mlContext)
        {
            // 1. Read in my dataset for training and testing
            IDataView trainData = mlContext.Data.LoadFromTextFile<SentimentData>(trainDataPath, hasHeader: false);

            /*
             * Split the dataset for model training and testing
             * testFraction --> 데이터 분할 백분율
             * TrainTestData --> struct 로 TestSet(IDataView), TrainSet(IDataView) 을 포함한다.
             * IDataView --> interface 쿼리 연산자 (변환)의 입력 및 출력 이것은 LINQ의 IEnumerable <T> 와 비슷한 기본 데이터 파이프 라인 유형 입니다.
             *               interface is the central concept of "data" in ML.NET.
             */

            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(trainData, testFraction: 0.2);

            return splitDataView;
        }


        /*
         * BuildAndTrainModel methods 설명
         * 1. 데이터를 추출하고 변환합니다.
         * 2. 모델을 학습시킵니다.
         * 3. 테스트 데이터를 기반으로 감정을 예측합니다.
         * 4. 모델을 반환합니다.
         */
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            /*
             * FeaturizeText(TransformsCatalog+TextTransforms, String, String)
             * var textPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text");
             */

            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            /*
             * Accuracy 메트릭은 테스트 세트에서 정확한 예측의 비율인 모델 정확도를 가져옵니다.
             * AreaUnderRocCurve 메트릭은 모델이 긍정 및 부정 클래스 분류의 정확도를 자신하는 정도를 표시합니다. AreaUnderRocCurve를 가능한 근접하게 하고자 합니다.
             * F1Score 메트릭은 모델의 F1 점수를 가져옵니다. 정밀도와 회수 사이의 균형을 측정한 값입니다. F1Score를 가능한 근접하게 하고자 합니다
             */

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                //SentimentText = "This was a very bad steak"
                SentimentText = "그건 좋아."
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    //SentimentText = "This was a horrible meal"
                    SentimentText = "그건 좋아."
                },
                new SentimentData
                {
                    //SentimentText = "I love this spaghetti."
                    SentimentText = "내건 나빠."
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }


        private static void runToTalML()
        {
            var ctx = new MLContext();

            // 1. Read in my dataset for training and testing

            IDataView trainData = ctx.Data.LoadFromTextFile<SentimentData>(trainDataPath, hasHeader: true);
            IDataView testData = ctx.Data.LoadFromTextFile<SentimentData>(testDataPath, hasHeader: true);

            // 2. Build an estimator pipline which transforms and add ML trainer
            IEstimator<ITransformer> est = ctx.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
                                              .Append(ctx.BinaryClassification.Trainers.LbfgsLogisticRegression("Label", "Features"));

            // 3. Train my model
            var model = est.Fit(trainData);

            // 4. Perform some predictions
            var predictions = model.Transform(testData);

            // 5. Evaluate the model
            var metrics = ctx.BinaryClassification.Evaluate(predictions, "Label", "Score");

            // 6. create Prediction Engine, predict on single instance of data
            var predictionEngine = ctx.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            var prediction = predictionEngine.Predict(new SentimentData
            {
                SentimentText = "Machine Learning is fun"
            });
        }

    }



    //public class Sentiment
    //{
    //    [LoadColumn(0)]
    //    public string Text { get; set; }

    //    [LoadColumn(1)]
    //    public bool Label { get; set; }

    //}

    //public class SentimentPrediction
    //{
    //    [ColumnName("PredictedLabel")]
    //    public bool Prediction { get; set; }

    //    public float Probabilty { get; set; }
    //    public float Score { get; set; }
    //}

    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }

}
