using Microsoft.ML.Data;

namespace MachineLearning.Models;

public class SentimentData
{
    [LoadColumn(0)]
    public string? Text { get; set; }
    
    [LoadColumn(1), ColumnName("Label")] 
    public bool Sentiment { get; set; }
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")] 
    public bool Prediction { get; set; }
    
    public float Probability { get; set; }
    public float Score { get; set; }
}