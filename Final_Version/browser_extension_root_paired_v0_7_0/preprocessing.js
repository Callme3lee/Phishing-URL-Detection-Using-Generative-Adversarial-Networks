(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.PhishingPreprocessing = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SPECIAL = { PAD: "<PAD>", UNK: "<UNK>", BOS: "<BOS>", EOS: "<EOS>" };
  const SUSPICIOUS = ["login", "verify", "account", "secure", "update", "bank", "signin", "password", "wallet"];

  function codePoints(value) { return Array.from(value); }
  function isDigit(character) { return /^\p{N}$/u.test(character); }
  function isLetter(character) { return /^\p{L}$/u.test(character); }
  function isAlnum(character) { return isDigit(character) || isLetter(character); }

  function decodeHtml(value, entities) {
    return value.replace(/&(#(?:x[0-9a-f]+|[0-9]+);?|[0-9a-z]+;?)/gi, (whole, name) => {
      if (name[0] === "#") {
        const hexadecimal = name[1]?.toLowerCase() === "x";
        const raw = name.slice(hexadecimal ? 2 : 1).replace(/;$/, "");
        const point = Number.parseInt(raw, hexadecimal ? 16 : 10);
        if (!Number.isFinite(point) || point < 0 || point > 0x10ffff) return "\ufffd";
        try { return String.fromCodePoint(point); } catch (_) { return "\ufffd"; }
      }
      const direct = entities[name];
      const alternate = name.endsWith(";") ? entities[name.slice(0, -1)] : entities[`${name};`];
      return direct ?? alternate ?? whole;
    });
  }

  function explicitAuthority(value, hasScheme) {
    let remainder = hasScheme ? value.replace(/^[A-Za-z][A-Za-z0-9+.-]*:\/\//, "") : value.replace(/^\/+/, "");
    const boundary = remainder.search(/[/?#]/);
    return boundary < 0 ? { authority: remainder, suffix: "" } : {
      authority: remainder.slice(0, boundary), suffix: remainder.slice(boundary),
    };
  }

  function explicitPort(authority) {
    const hostPort = authority.slice(authority.lastIndexOf("@") + 1);
    if (hostPort.startsWith("[")) {
      const close = hostPort.indexOf("]");
      return close >= 0 && hostPort[close + 1] === ":" ? hostPort.slice(close + 2) : "";
    }
    const colon = hostPort.lastIndexOf(":");
    return colon >= 0 && /^\d+$/.test(hostPort.slice(colon + 1)) ? hostPort.slice(colon + 1) : "";
  }

  function canonicalizeUrl(raw, entities, maximumLength = 4096) {
    let value = decodeHtml(String(raw), entities).trim();
    if (!value) throw new Error("empty_url");
    if (value.length > maximumLength) throw new Error("raw_url_too_long");
    if (codePoints(value).some((character) => character.codePointAt(0) < 32)) throw new Error("control_character");
    const hasScheme = /^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(value);
    const parseValue = hasScheme ? value : `http://${value.replace(/^\/+/, "")}`;
    let parsed;
    try { parsed = new URL(parseValue); } catch (_) { throw new Error("url_parse_error"); }
    let hostname = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.+$/, "");
    if (!hostname || hostname.length > 253) throw new Error("invalid_hostname");
    const outer = explicitAuthority(value, hasScheme);
    const port = explicitPort(outer.authority);
    if (port && Number(port) > 65535) throw new Error("invalid_port");
    const displayHost = hostname.includes(":") ? `[${hostname}]` : hostname;
    const netloc = `${displayHost}${port ? `:${port}` : ""}`;
    const hasPath = outer.suffix.startsWith("/");
    const path = hasPath ? parsed.pathname : "";
    const query = parsed.search ? parsed.search.slice(1) : "";
    const fragment = parsed.hash ? parsed.hash.slice(1) : "";
    const canonical = `${netloc}${path}${query ? `?${query}` : ""}${fragment ? `#${fragment}` : ""}`;
    return { canonicalUrl: canonical, hostname, scheme: parsed.protocol.slice(0, -1).toLowerCase() };
  }

  function ipv4(hostname) {
    const parts = hostname.split(".");
    return parts.length === 4 && parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255);
  }
  function ipHostname(hostname) { return ipv4(hostname) || hostname.includes(":"); }

  function createPslExtractor(rules) {
    const exact = new Set(); const wildcard = new Set(); const exceptions = new Set();
    for (const rule of rules) {
      if (rule.startsWith("!")) exceptions.add(rule.slice(1));
      else if (rule.startsWith("*.")) wildcard.add(rule.slice(2));
      else exact.add(rule);
    }
    return function extract(hostname) {
      if (ipv4(hostname)) return { subdomain: "", domain: hostname, suffix: "" };
      const labels = hostname.split(".").filter(Boolean);
      if (!labels.length) return { subdomain: "", domain: "", suffix: "" };
      let suffixStart = labels.length;
      for (let index = 0; index < labels.length; index += 1) {
        const candidate = labels.slice(index).join(".");
        if (exceptions.has(candidate)) { suffixStart = index + 1; break; }
      }
      if (suffixStart === labels.length) {
        let bestLength = 0;
        for (let index = 0; index < labels.length; index += 1) {
          const candidate = labels.slice(index).join(".");
          const length = labels.length - index;
          if (exact.has(candidate) && length > bestLength) { suffixStart = index; bestLength = length; }
          if (index > 0 && wildcard.has(candidate) && length + 1 > bestLength) {
            suffixStart = index - 1; bestLength = length + 1;
          }
        }
      }
      const suffix = suffixStart < labels.length ? labels.slice(suffixStart).join(".") : "";
      const domainIndex = suffixStart - 1;
      const domain = domainIndex >= 0 ? labels[domainIndex] : "";
      const subdomain = domainIndex > 0 ? labels.slice(0, domainIndex).join(".") : "";
      return { subdomain, domain, suffix };
    };
  }

  function entropy(value) {
    const characters = codePoints(value);
    if (!characters.length) return 0;
    const counts = new Map();
    for (const character of characters) counts.set(character, (counts.get(character) ?? 0) + 1);
    let result = 0;
    for (const count of counts.values()) { const p = count / characters.length; result -= p * Math.log2(p); }
    return result;
  }

  function ratioCount(value, predicate) {
    const characters = codePoints(value);
    return characters.filter(predicate).length / Math.max(1, characters.length);
  }

  function maximumRunRatio(value) {
    const characters = codePoints(value);
    if (!characters.length) return 0;
    let maximum = 1; let current = 1;
    for (let index = 1; index < characters.length; index += 1) {
      current = characters[index] === characters[index - 1] ? current + 1 : 1;
      maximum = Math.max(maximum, current);
    }
    return maximum / characters.length;
  }

  function countOccurrences(value, token) {
    if (!token) return 0;
    let count = 0; let position = 0;
    while ((position = value.indexOf(token, position)) >= 0) { count += 1; position += token.length; }
    return count;
  }

  function parseCanonical(url) {
    const parsed = new URL(`http://${url.replace(/^\/+/, "")}`);
    let hostname = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "");
    const authority = url.split(/[/?#]/, 1)[0];
    return {
      hostname, path: parsed.pathname === "/" && !url.slice(authority.length).startsWith("/") ? "" : parsed.pathname,
      query: parsed.search ? parsed.search.slice(1) : "", explicitPort: Boolean(explicitPort(authority)),
    };
  }

  function lexicalFeatures(url, extractPsl) {
    const parsed = parseCanonical(url);
    const hostname = parsed.hostname; const path = parsed.path; const query = parsed.query;
    const extracted = extractPsl(hostname);
    const registered = [extracted.domain, extracted.suffix].filter(Boolean).join(".");
    const subdomain = extracted.subdomain || ""; const total = codePoints(url).length;
    const lower = url.toLowerCase();
    return Float32Array.from([
      Math.log1p(total), Math.log1p(codePoints(hostname).length), Math.log1p(codePoints(registered).length),
      Math.log1p(codePoints(extracted.domain || "").length), Math.log1p(codePoints(extracted.suffix || "").length),
      Math.log1p(codePoints(subdomain).length), Math.log1p(codePoints(path).length), Math.log1p(codePoints(query).length),
      codePoints(hostname).length / Math.max(1, total), codePoints(path).length / Math.max(1, total),
      codePoints(query).length / Math.max(1, total), ratioCount(url, isDigit), ratioCount(url, (c) => !isAlnum(c)),
      ratioCount(hostname, isDigit), countOccurrences(hostname, "-") / Math.max(1, codePoints(hostname).length),
      countOccurrences(hostname, ".") / Math.max(1, codePoints(hostname).length), ratioCount(path, isDigit),
      ratioCount(path, (c) => !isAlnum(c)), ratioCount(query, isDigit), ratioCount(query, (c) => !isAlnum(c)),
      subdomain.split(".").filter(Boolean).length, path.split("/").filter(Boolean).length,
      query.split(/[&;]/).filter(Boolean).length, entropy(hostname), entropy(path), entropy(query),
      ipHostname(hostname) ? 1 : 0, hostname.includes("xn--") ? 1 : 0, extracted.suffix ? 1 : 0,
      countOccurrences(url, "%") / Math.max(1, total),
      SUSPICIOUS.reduce((sum, token) => sum + countOccurrences(lower, token), 0) / Math.max(1, total),
      maximumRunRatio(url), countOccurrences(url, "=") / Math.max(1, total),
      countOccurrences(url, "&") / Math.max(1, total), parsed.explicitPort ? 1 : 0,
    ]);
  }

  function diagnosticSignals(canonicalUrl, rawLexical, scaledLexical, featureNames) {
    const index = Object.fromEntries(featureNames.map((name, position) => [name, position]));
    const raw = (name) => Number(rawLexical[index[name]] ?? 0);
    const deviation = (name) => Number(scaledLexical[index[name]] ?? 0);
    const urlLength = Math.max(0, Math.round(Math.expm1(raw("log_url_length"))));
    const queryLength = Math.max(0, Math.round(Math.expm1(raw("log_query_length"))));
    const lower = canonicalUrl.toLowerCase();
    const matchedTerms = SUSPICIOUS.filter((term) => lower.includes(term));
    const signals = [];
    const add = (code, strength, values = {}) => signals.push({ code, strength, values });

    // Scaled values use the frozen training median and IQR. A value of 1.5
    // therefore means the feature is substantially above its training median;
    // it is diagnostic evidence, not a causal attribution of the neural score.
    if (raw("ip_hostname") >= 0.5) add("ip_hostname", 10);
    if (raw("punycode_hostname") >= 0.5) add("punycode_hostname", 9);
    if (raw("explicit_port") >= 0.5) add("explicit_port", 7);
    if (matchedTerms.length) add("suspicious_terms", 5 + matchedTerms.length,
      { words: matchedTerms.slice(0, 4).join(", ") });
    if (deviation("log_url_length") >= 1.5) add("long_url", deviation("log_url_length"), { count: urlLength });
    if (deviation("url_special_ratio") >= 1.5) add("many_symbols", deviation("url_special_ratio"));
    if (Math.max(deviation("url_digit_ratio"), deviation("hostname_digit_ratio")) >= 1.5) {
      add("many_digits", Math.max(deviation("url_digit_ratio"), deviation("hostname_digit_ratio")));
    }
    if (raw("subdomain_label_count") >= 3 || deviation("subdomain_label_count") >= 1.5) {
      add("deep_subdomains", Math.max(2, deviation("subdomain_label_count")),
        { count: Math.round(raw("subdomain_label_count")) });
    }
    if (raw("path_depth") >= 5 || deviation("path_depth") >= 1.5) {
      add("deep_path", Math.max(2, deviation("path_depth")), { count: Math.round(raw("path_depth")) });
    }
    if (queryLength >= 80 || deviation("log_query_length") >= 1.5 || raw("query_parameter_count") >= 5) {
      add("complex_query", Math.max(2, deviation("log_query_length")), { count: queryLength });
    }
    if (deviation("hostname_entropy") >= 2) add("hostname_entropy", deviation("hostname_entropy"));
    if (raw("percent_encoding_ratio") > 0 && deviation("percent_encoding_ratio") >= 1) {
      add("encoded_characters", Math.max(2, deviation("percent_encoding_ratio")));
    }
    if (deviation("maximum_character_run_ratio") >= 2) add("repeated_characters", deviation("maximum_character_run_ratio"));
    if (raw("valid_public_suffix") < 0.5 && raw("ip_hostname") < 0.5) add("unrecognized_suffix", 4);
    return signals.sort((left, right) => right.strength - left.strength).slice(0, 4);
  }

  function encodeUrl(url, vocabulary, maximumLength = 128) {
    const output = new BigInt64Array(maximumLength);
    const characters = codePoints(url).slice(0, maximumLength - 1);
    const unknown = vocabulary[SPECIAL.UNK];
    characters.forEach((character, index) => { output[index] = BigInt(vocabulary[character] ?? unknown); });
    output[characters.length] = BigInt(vocabulary[SPECIAL.EOS]);
    return output;
  }

  function createPreprocessor(resources) {
    const extractPsl = createPslExtractor(resources.pslRules.rules);
    const center = Float32Array.from(resources.scaler.center);
    const scale = Float32Array.from(resources.scaler.scale, (value) => value === 0 ? 1 : value);
    return {
      canonicalize(raw) { return canonicalizeUrl(raw, resources.htmlEntities); },
      prepare(raw) {
        const canonical = canonicalizeUrl(raw, resources.htmlEntities);
        const rawLexical = lexicalFeatures(canonical.canonicalUrl, extractPsl);
        const lexical = Float32Array.from(rawLexical, (value, index) => (value - center[index]) / scale[index]);
        const tokens = encodeUrl(canonical.canonicalUrl, resources.vocabulary, 128);
        const signals = diagnosticSignals(canonical.canonicalUrl, rawLexical, lexical, resources.scaler.feature_names);
        return { ...canonical, tokens, rawLexical, lexical, signals };
      },
      extractPsl,
    };
  }

  return { createPreprocessor, canonicalizeUrl, createPslExtractor, lexicalFeatures, diagnosticSignals, encodeUrl };
});
