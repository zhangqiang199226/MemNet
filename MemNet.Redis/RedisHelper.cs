using System;
using StackExchange.Redis;

namespace MemNet.Redis;

class RedisHelper
{
    public static bool IsVectorSupported(IConnectionMultiplexer redis)
    {
        var endpoints = redis.GetEndPoints();
        var server = redis.GetServer(endpoints[0]);
        var result = server.Execute("MODULE", "LIST");
	
        if (result.IsNull)
        {
            return false;
        }

        var modules = (RedisResult[]?)result;
        if (modules == null || modules.Length == 0)
        {
            return false;
        }
	
        bool hasVectorSupport = false;
	
        foreach (var module in modules)
        {
            var items = (RedisResult[]?)module;
            if (items == null) continue;
		
            string? name = null;
            string? version = null;
		
            for (int i = 0; i < items.Length - 1; i += 2)
            {
                var key = items[i].ToString() ?? "";
                var val = items[i + 1].ToString() ?? "";
			
                if (string.Equals(key, "name", StringComparison.OrdinalIgnoreCase))
                    name = val;
                else if (string.Equals(key, "ver", StringComparison.OrdinalIgnoreCase))
                    version = val;
            }

            if (!string.IsNullOrEmpty(name))
            {
                var nameLower = name?.ToLowerInvariant() ?? "";
                if (nameLower.Contains("search") || 
                    nameLower.Contains("vector") || 
                    nameLower.Contains("vecsim") ||
                    nameLower == "ft")
                {
                    hasVectorSupport = true;
                }
            }
        }
        return hasVectorSupport;
    }
}