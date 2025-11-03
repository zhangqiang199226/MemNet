using System;
using MemNet.Abstractions;
using Microsoft.Extensions.DependencyInjection;
using StackExchange.Redis;

namespace MemNet.Redis;

/// <summary>
/// MemNet.Redis service registration extensions
/// </summary>
public static class RedisServiceCollectionExtensions
{
    /// <summary>
    /// Add MemNet with Redis vector store support
    /// </summary>
    /// <param name="services">Service collection</param>
    /// <param name="connectionString">Redis connection string</param>
    /// <param name="configureOptions">Optional Redis configuration options</param>
    public static IServiceCollection WithMemNetRedis(
        this IServiceCollection services,
        string connectionString,
        Action<ConfigurationOptions>? configureOptions = null)
    {
        if (string.IsNullOrEmpty(connectionString))
        {
            throw new ArgumentNullException(nameof(connectionString));
        }

        // Configure Redis connection
        var options = ConfigurationOptions.Parse(connectionString);
        configureOptions?.Invoke(options);

        // Register IConnectionMultiplexer
        services.AddSingleton<IConnectionMultiplexer>(_ => ConnectionMultiplexer.Connect(options));
        // Register RedisVectorStore
        services.AddSingleton<IVectorStore, RedisVectorStore>();

        return services;
    }
}

