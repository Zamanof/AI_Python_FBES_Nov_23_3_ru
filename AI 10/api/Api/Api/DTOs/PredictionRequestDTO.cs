using System.ComponentModel.DataAnnotations;

namespace Api.DTOs;

public class PredictionRequestDTO
{
    [Required]
    [Range(0, double.MaxValue, ErrorMessage ="Bedrooms must be a positive number")]
    public float Bedrooms { get; set; }

    [Required]
    [Range(0, double.MaxValue, ErrorMessage = "Bathrooms must be a positive number")]
    public float Bathrooms { get; set; }

    [Required]
    [Range(0, double.MaxValue, ErrorMessage = "Square meters must be a positive number")]
    public float Sqm { get; set; }

    [Required]
    [MinLength(1, ErrorMessage ="City is required")]
    public string City { get; set; } = string.Empty;
}
