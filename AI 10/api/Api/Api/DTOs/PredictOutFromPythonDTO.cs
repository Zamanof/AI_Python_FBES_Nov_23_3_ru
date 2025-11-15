using System.Text.Json.Serialization;

namespace Api.DTOs;

public class PredictOutFromPythonDTO
{
    [JsonPropertyName("priceAzn")]
    public double PriceAZN { get; set; }
}
