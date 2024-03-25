namespace MachineLearning.MLContexts
{
    public class BinaryClassificationVS
    {
        public static void Run()
        {
            //Load sample data
            var sampleData = new SentimentModel.ModelInput()
            {
                Col0 = @"Crust is not good.",
            };

            //Load model and predict output
            var result = SentimentModel.Predict(sampleData);

        }
    }
}
