using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using MemNet.Abstractions;
using MemNet.Config;
using MemNet.Models;
using Microsoft.Extensions.Options;

namespace MemNet.VectorStores;

/// <summary>
/// Milvus vector store implementation
/// </summary>
public class MilvusVectorStore : IVectorStore
{
    private readonly HttpClient _httpClient;
    private readonly VectorStoreConfig _config;
    private readonly string _collectionName;

    public MilvusVectorStore(HttpClient httpClient, IOptions<MemoryConfig> config)
    {
        _httpClient = httpClient;
        _config = config.Value.VectorStore;
        _collectionName = _config.CollectionName;

        // Configure HttpClient
        if (_httpClient.BaseAddress == null)
        {
            _httpClient.BaseAddress = new Uri($"http://{_config.Host}:{_config.Port}");
        }

        if (!string.IsNullOrEmpty(_config.ApiKey))
        {
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {_config.ApiKey}");
        }

        // Ensure collection exists
        EnsureCollectionAsync().GetAwaiter().GetResult();
    }

    private async Task EnsureCollectionAsync()
    {
        try
        {
            // Check if collection exists
            var checkRequest = new
            {
                collection_name = _collectionName
            };

            var checkResponse = await _httpClient.PostAsJsonAsync("/v1/vector/collections/describe", checkRequest);
            
            if (checkResponse.IsSuccessStatusCode)
            {
                return;
            }

            // Create collection
            var createRequest = new
            {
                collection_name = _collectionName,
                dimension = 1536,
                metric_type = "COSINE",
                primary_field = "id",
                vector_field = "embedding"
            };

            await _httpClient.PostAsJsonAsync("/v1/vector/collections/create", createRequest);
        }
        catch
        {
            // Collection might already exist or Milvus is not available
        }
    }

    public async Task InsertAsync(List<MemoryItem> memories, CancellationToken ct = default)
    {
        var data = memories.Select(m => new Dictionary<string, object>
        {
            ["id"] = m.Id,
            ["embedding"] = m.Embedding,
            ["data"] = m.Data,
            ["userId"] = m.UserId ?? string.Empty,
            ["agentId"] = m.AgentId ?? string.Empty,
            ["runId"] = m.RunId ?? string.Empty,
            ["metadata"] = System.Text.Json.JsonSerializer.Serialize(m.Metadata),
            ["createdAt"] = m.CreatedAt.ToString("o"),
            ["updatedAt"] = m.UpdatedAt?.ToString("o") ?? string.Empty,
            ["hash"] = m.Hash ?? string.Empty
        }).ToList();

        var request = new
        {
            collection_name = _collectionName,
            data
        };

        var response = await _httpClient.PostAsJsonAsync("/v1/vector/insert", request, ct);
        response.EnsureSuccessStatusCode();
    }

    public async Task UpdateAsync(List<MemoryItem> memories, CancellationToken ct = default)
    {
        // Delete existing entries first
        var ids = memories.Select(m => m.Id).ToList();
        await DeleteMultipleAsync(ids, ct);

        // Insert updated memories
        await InsertAsync(memories, ct);
    }

    public async Task<List<MemorySearchResult>> SearchAsync(float[] queryVector, string? userId = null, int limit = 100, CancellationToken ct = default)
    {
        var searchRequest = new
        {
            collection_name = _collectionName,
            vector = queryVector,
            limit,
            output_fields = new[] { "id", "data", "userId", "agentId", "runId", "metadata", "createdAt", "updatedAt", "hash" },
            filter = userId != null ? $"userId == \"{userId}\"" : null
        };

        var response = await _httpClient.PostAsJsonAsync("/v1/vector/search", searchRequest, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<MilvusSearchResponse>(ct);

        if (result?.Data == null || result.Data.Count == 0)
        {
            return new List<MemorySearchResult>();
        }

        return result.Data[0].Select(item => new MemorySearchResult
        {
            Id = item.Id,
            Memory = new MemoryItem
            {
                Id = item.Id,
                Data = item.Data,
                Embedding = Array.Empty<float>(),
                UserId = item.UserId,
                AgentId = item.AgentId,
                RunId = item.RunId,
                Metadata = !string.IsNullOrEmpty(item.Metadata) 
                    ? System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(item.Metadata) ?? new Dictionary<string, object>()
                    : new Dictionary<string, object>(),
                CreatedAt = DateTime.Parse(item.CreatedAt),
                UpdatedAt = !string.IsNullOrEmpty(item.UpdatedAt) ? DateTime.Parse(item.UpdatedAt) : null,
                Hash = item.Hash
            },
            Score = item.Distance
        }).ToList();
    }

    public async Task<List<MemoryItem>> ListAsync(string? userId = null, int limit = 100, CancellationToken ct = default)
    {
        var queryRequest = new
        {
            collection_name = _collectionName,
            filter = userId != null ? $"userId == \"{userId}\"" : string.Empty,
            limit,
            output_fields = new[] { "id", "data", "userId", "agentId", "runId", "metadata", "createdAt", "updatedAt", "hash" }
        };

        var response = await _httpClient.PostAsJsonAsync("/v1/vector/query", queryRequest, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<MilvusQueryResponse>(ct);

        return result?.Data?.Select(item => new MemoryItem
        {
            Id = item.Id,
            Data = item.Data,
            Embedding = Array.Empty<float>(),
            UserId = item.UserId,
            AgentId = item.AgentId,
            RunId = item.RunId,
            Metadata = !string.IsNullOrEmpty(item.Metadata)
                ? System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(item.Metadata) ?? new Dictionary<string, object>()
                : new Dictionary<string, object>(),
            CreatedAt = DateTime.Parse(item.CreatedAt),
            UpdatedAt = !string.IsNullOrEmpty(item.UpdatedAt) ? DateTime.Parse(item.UpdatedAt) : null,
            Hash = item.Hash
        }).ToList() ?? new List<MemoryItem>();
    }

    public async Task<MemoryItem?> GetAsync(string memoryId, CancellationToken ct = default)
    {
        var queryRequest = new
        {
            collection_name = _collectionName,
            filter = $"id == \"{memoryId}\"",
            limit = 1,
            output_fields = new[] { "id", "embedding", "data", "userId", "agentId", "runId", "metadata", "createdAt", "updatedAt", "hash" }
        };

        var response = await _httpClient.PostAsJsonAsync("/v1/vector/query", queryRequest, ct);
        
        if (!response.IsSuccessStatusCode)
        {
            return null;
        }

        var result = await response.Content.ReadFromJsonAsync<MilvusQueryResponse>(ct);

        var item = result?.Data?.FirstOrDefault();
        if (item == null)
        {
            return null;
        }

        return new MemoryItem
        {
            Id = item.Id,
            Data = item.Data,
            Embedding = item.Embedding ?? Array.Empty<float>(),
            UserId = item.UserId,
            AgentId = item.AgentId,
            RunId = item.RunId,
            Metadata = !string.IsNullOrEmpty(item.Metadata)
                ? System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(item.Metadata) ?? new Dictionary<string, object>()
                : new Dictionary<string, object>(),
            CreatedAt = DateTime.Parse(item.CreatedAt),
            UpdatedAt = !string.IsNullOrEmpty(item.UpdatedAt) ? DateTime.Parse(item.UpdatedAt) : null,
            Hash = item.Hash
        };
    }

    public async Task DeleteAsync(string memoryId, CancellationToken ct = default)
    {
        await DeleteMultipleAsync(new List<string> { memoryId }, ct);
    }

    public async Task DeleteByUserAsync(string userId, CancellationToken ct = default)
    {
        var deleteRequest = new
        {
            collection_name = _collectionName,
            filter = $"userId == \"{userId}\""
        };

        var response = await _httpClient.PostAsJsonAsync("/v1/vector/delete", deleteRequest, ct);
        response.EnsureSuccessStatusCode();
    }

    private async Task DeleteMultipleAsync(List<string> ids, CancellationToken ct = default)
    {
        var deleteRequest = new
        {
            collection_name = _collectionName,
            filter = $"id in [{string.Join(",", ids.Select(id => $"\"{id}\""))}]"
        };

        var response = await _httpClient.PostAsJsonAsync("/v1/vector/delete", deleteRequest, ct);
        response.EnsureSuccessStatusCode();
    }

    // Internal classes for JSON deserialization
    private class MilvusSearchResponse
    {
        [JsonPropertyName("code")]
        public int Code { get; set; }

        [JsonPropertyName("data")]
        public List<List<MilvusSearchItem>>? Data { get; set; }
    }

    private class MilvusSearchItem
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("distance")]
        public float Distance { get; set; }

        [JsonPropertyName("data")]
        public string Data { get; set; } = string.Empty;

        [JsonPropertyName("userId")]
        public string? UserId { get; set; }

        [JsonPropertyName("agentId")]
        public string? AgentId { get; set; }

        [JsonPropertyName("runId")]
        public string? RunId { get; set; }

        [JsonPropertyName("metadata")]
        public string Metadata { get; set; } = string.Empty;

        [JsonPropertyName("createdAt")]
        public string CreatedAt { get; set; } = string.Empty;

        [JsonPropertyName("updatedAt")]
        public string? UpdatedAt { get; set; }

        [JsonPropertyName("hash")]
        public string? Hash { get; set; }
    }

    private class MilvusQueryResponse
    {
        [JsonPropertyName("code")]
        public int Code { get; set; }

        [JsonPropertyName("data")]
        public List<MilvusQueryItem>? Data { get; set; }
    }

    private class MilvusQueryItem
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("embedding")]
        public float[]? Embedding { get; set; }

        [JsonPropertyName("data")]
        public string Data { get; set; } = string.Empty;

        [JsonPropertyName("userId")]
        public string? UserId { get; set; }

        [JsonPropertyName("agentId")]
        public string? AgentId { get; set; }

        [JsonPropertyName("runId")]
        public string? RunId { get; set; }

        [JsonPropertyName("metadata")]
        public string Metadata { get; set; } = string.Empty;

        [JsonPropertyName("createdAt")]
        public string CreatedAt { get; set; } = string.Empty;

        [JsonPropertyName("updatedAt")]
        public string? UpdatedAt { get; set; }

        [JsonPropertyName("hash")]
        public string? Hash { get; set; }
    }
}

