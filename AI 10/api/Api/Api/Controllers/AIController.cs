using Api.DTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace Api.Controllers;

[Route("api/[controller]")]
[ApiController]
public class AIController : ControllerBase
{
    private readonly HttpClient _http;

    public AIController(IHttpClientFactory factory) => _http = factory.CreateClient("ai");

    [HttpPost("predict")]
    public async Task<IActionResult> Predict([FromBody] PredictIn dto)
    {
        var res = await _http.PostAsJsonAsync("/predict", dto);

        if (!res.IsSuccessStatusCode)
        {
            return StatusCode((int)res.StatusCode, await res.Content.ReadAsStringAsync());
        }

        var body = await res.Content.ReadFromJsonAsync<PredictOutPython>();
        return Ok(body);
    }
}
