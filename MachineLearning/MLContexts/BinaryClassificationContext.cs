using MachineLearning.Models;
using Microsoft.ML;
using Spectre.Console;

namespace MachineLearning.MLContexts;

public class BinaryClassificationContext
{
    public static void Run()
    {
        var context = new MLContext();
        
        // Load data
        var data = context.Data.LoadFromTextFile<SentimentData>("YelpReviewData.txt");
        
        // Separate data into test set (20%)
        var splitDataView = context.Data.TrainTestSplit(data, testFraction: 0.2);
        
        // Build model using binary classification because we are predicting sentiment (reviews are positive or negative)
        var estimator = context.Transforms.Text
            .FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName: "Features"));

        ITransformer model = default!;
        var rule = new Rule("Create and Train Model");
        AnsiConsole.Live(rule)
            .Start(console =>
            {
                //Train model
                model = estimator.Fit(splitDataView.TrainSet);
                var predictions = model.Transform(splitDataView.TestSet);
                
                rule.Title = "🏴 Training Complete, Evaluating Accuracy.";
                console.Refresh();
                
                //Evaluate model accuracy
                var metrics = context.BinaryClassification.Evaluate(predictions);

                var table = new Table()
                    .MinimalBorder()
                    .Title("💯 Model Accuracy");
                table.AddColumns("Accuracy", "AUC", "F1 Score");
                table.AddRow($"{metrics.Accuracy:P2}", $"{metrics.AreaUnderRocCurve:P2}", $"{metrics.F1Score:P2}");
                
                console.UpdateTarget(table);
                console.Refresh();
            });
        
        // Predict sentiment (positive or negative) from user input (review text)
        var keepGoing = true;
        while (keepGoing)
        {
            var text = AnsiConsole.Ask<string>("What's your [green]review text[/]?");
            if(text == "exit")
            {
                keepGoing = false;
                continue;
            }
            
            var engine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var input = new SentimentData { Text = text };
            var result = engine.Predict(input);
            var style = result.Prediction
                ? (color: "green", emoji: "+1")
                : (color: "red", emoji: "-1");
            AnsiConsole.MarkupLine($"{style.emoji} [{style.color}]\"{text}\" ({result.Probability:P00})[/] ");
        }
        
        // Save model
        //context.Model.Save(model, data.Schema, "sentiment-model.zip");
        
        // Load model
        //var loadedModel = context.Model.Load("sentiment-model.zip", out var schema);
    }
}