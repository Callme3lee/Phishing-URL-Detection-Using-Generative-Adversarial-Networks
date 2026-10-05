(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PhishingBrowserPolicy = api;
})(globalThis, function () {
  "use strict";

  // Keep the catalogue API identifier stable inside extension v0.6.0 so an
  // already-running Manifest V3 worker cannot reject the updated JSON during
  // a Chrome reload. Measurement schema 7 records the Google Docs revision.
  const POLICY_VERSION = "reviewed_site_catalog_v3";

  function servicePage(rawUrl, catalog) {
    try {
      // Reject ambiguous input before the URL parser can normalise it.
      if (/[\\\u0000-\u0020\u007f]/.test(rawUrl)) return null;
      const url = new URL(rawUrl);
      if (url.protocol !== "https:" || url.username || url.password || url.port) return null;
      const host = url.hostname.toLowerCase();
      if (!/^[a-z0-9.-]+$/.test(host) || host.endsWith(".")) return null;
      const path = decodeURIComponent(url.pathname);
      if (/[\\\u0000-\u001f\u007f]/.test(path) || path.includes("//")) return null;
      if (!catalog || catalog.policy_version !== POLICY_VERSION || !Array.isArray(catalog.reviewed_rules)) return null;
      const globalExcluded = new RegExp(catalog.global_excluded_path_pattern, "i");
      if (globalExcluded.test(path)) return null;
      const rule = catalog.reviewed_rules.find(entry => entry.hosts.includes(host)
        && new RegExp(entry.allowed_path_pattern).test(path)
        && !(entry.excluded_path_pattern && new RegExp(entry.excluded_path_pattern, "i").test(path)));
      return rule ? { ruleId: rule.id, host, registrableDomain: rule.registrable_domain,
        popularityRank: rule.tranco_rank ?? null, policyVersion: POLICY_VERSION } : null;
    } catch (_) { return null; }
  }

  function userControlledServiceContent(rawUrl, service) {
    if (service?.ruleId !== "google-workspace") return false;
    try {
      const url = new URL(rawUrl);
      const host = url.hostname.toLowerCase();
      const path = decodeURIComponent(url.pathname);
      if (host === "docs.google.com") {
        return /^\/(?:document|spreadsheets|presentation|forms|drawings)(?:\/|$)/i.test(path);
      }
      if (host === "drive.google.com") {
        return /^\/(?:file(?:\/|$)|open(?:\/|$)|uc(?:\/|$)|drive\/(?:u\/\d+\/)?folders(?:\/|$))/i.test(path);
      }
      if (["sheets.google.com", "slides.google.com", "forms.google.com", "sites.google.com"].includes(host)) {
        return path !== "/" && path !== "";
      }
      return false;
    } catch (_) { return false; }
  }

  function googleDocumentQueryNoise(rawUrl, service, result) {
    if (service?.ruleId !== "google-workspace" || !result?.queryView?.applied) return false;
    if (!["allow", "secondary_review"].includes(result.queryView.tier)) return false;
    try {
      const url = new URL(rawUrl);
      if (url.hostname.toLowerCase() !== "docs.google.com" || !url.search) return false;
      const path = decodeURIComponent(url.pathname);
      // Restrict this correction to ordinary collaborative document routes.
      // Public forms, published presentations, Sites and Drive shares retain
      // the normal risk-preserving shared-content policy.
      if (!/^\/(?:document|spreadsheets|presentation|drawings)\/(?:u\/\d+\/)?d\/(?!e(?:\/|$))[^/]+\/(?:edit|view|preview|copy|mobilebasic)(?:\/|$)/i.test(path)) return false;
      const safeKeys = new Set([
        "chrome", "embedded", "gid", "headers", "pli", "range", "resourcekey",
        "rm", "single", "tab", "usp", "widget",
      ]);
      const keys = [...url.searchParams.keys()].map(value => value.toLowerCase());
      return keys.length > 0 && keys.length <= 12 && keys.every(key => safeKeys.has(key));
    } catch (_) { return false; }
  }

  function googleCollaborativeDocumentRoute(rawUrl, service) {
    if (service?.ruleId !== "google-workspace") return false;
    try {
      const url = new URL(rawUrl);
      if (url.hostname.toLowerCase() !== "docs.google.com") return false;
      const path = decodeURIComponent(url.pathname);
      if (!/^\/(?:document|spreadsheets|presentation|drawings)\/(?:u\/\d+\/)?d\/(?!e(?:\/|$))[^/]+\/(?:edit|view|preview|copy|mobilebasic)(?:\/|$)/i.test(path)) return false;
      const safeKeys = new Set([
        "chrome", "embedded", "gid", "headers", "pli", "range", "resourcekey",
        "rm", "single", "tab", "usp", "widget",
      ]);
      const keys = [...url.searchParams.keys()].map(value => value.toLowerCase());
      return keys.length <= 12 && keys.every(key => safeKeys.has(key));
    } catch (_) { return false; }
  }

  function sharedContentDecision(result, service, mode, continuedOnce = false) {
    const modelTier = result?.tier ?? "unknown";
    if (googleDocumentQueryNoise(result?.rawUrl ?? "", service, result)) {
      return {
        modelTier, effectiveTier: "secondary_review", interrupt: false, mode,
        continuedOnce: false, reason: "reviewed_google_document_query_noise",
        serviceRule: service.ruleId, policyVersion: POLICY_VERSION,
        popularityRank: service.popularityRank, queryAdjusted: true,
        headline: "Shared content — review suggested",
        explanation: "This is an official Google document address. Its changing query values raised the full URL score, while the same address without those values did not. The page remains available; verify who shared it before entering sensitive information.",
      };
    }
    if (googleCollaborativeDocumentRoute(result?.rawUrl ?? "", service)) {
      const effectiveTier = {
        allow: "allow", secondary_review: "secondary_review", caution: "secondary_review",
        warning: "caution", block: "warning", unknown: "secondary_review",
      }[modelTier] ?? "secondary_review";
      return {
        modelTier, effectiveTier, interrupt: false, mode, continuedOnce: false,
        reason: "reviewed_google_document", serviceRule: service.ruleId,
        policyVersion: POLICY_VERSION, popularityRank: service.popularityRank,
        headline: effectiveTier === "allow" ? "Recognised shared content"
          : effectiveTier === "secondary_review" ? "Shared content — review suggested"
          : effectiveTier === "caution" ? "Shared content — use caution"
          : "Shared content — high risk",
        explanation: effectiveTier === "allow"
          ? "This is an official Google document address, and the URL model did not find a strong phishing pattern."
          : "This is an official Google document editor or viewer address. It remains available to avoid disrupting normal work, but the model result is preserved; verify who shared the document before trusting its contents or links.",
      };
    }
    const effectiveTier = {
      allow: "allow",
      secondary_review: "secondary_review",
      caution: "secondary_review",
      warning: "warning",
      block: "block",
      unknown: "secondary_review",
    }[modelTier] ?? "secondary_review";
    const presentation = {
      allow: ["Recognised shared content", "This page is hosted by Google, and the URL model did not find a strong phishing pattern."],
      secondary_review: ["Shared content — review suggested", "This page is hosted by Google but was created or shared by a user. Verify who sent it before entering sensitive information."],
      caution: ["Shared content — use caution", "This user-created Google link has a high URL risk score. The page remains available, but verify the sender before continuing."],
      warning: ["Shared content — high risk", "This user-created Google link has a high URL risk score. Navigation is paused in Normal mode, but you can choose to continue."],
      block: ["Shared content — very high risk", "This user-created Google link reached the highest URL risk range. Navigation is paused in Normal mode, but you can choose to continue."],
    }[effectiveTier];
    const blockingTier = effectiveTier === "warning" || effectiveTier === "block";
    const continuedBlock = continuedOnce && blockingTier;
    const passiveBlock = mode === "passive" && blockingTier;
    return {
      modelTier, effectiveTier, interrupt: mode === "normal" && blockingTier && !continuedBlock, mode,
      continuedOnce: continuedBlock,
      reason: "reviewed_user_content", serviceRule: service.ruleId, policyVersion: POLICY_VERSION,
      popularityRank: service.popularityRank,
      headline: continuedBlock
        ? `${effectiveTier === "block" ? "Very high-risk" : "High-risk"} shared content — continued once`
        : passiveBlock
        ? `${effectiveTier === "block" ? "Very high-risk" : "High-risk"} shared content — passive mode`
        : presentation[0],
      explanation: continuedBlock
        ? "You chose to continue once. The warning remains visible, so avoid entering sensitive information unless you trust the destination."
        : passiveBlock
        ? "Passive mode keeps this high-risk shared page available and shows a warning instead of interrupting navigation."
        : presentation[1],
    };
  }

  function queryView(rawUrl) {
    const url = new URL(rawUrl);
    const parts = url.search.slice(1).split("&");
    const count = url.search ? parts.filter(Boolean).length : 0;
    let removed = 0;
    const clean = parts.map(part => {
      const equals = part.indexOf("=");
      if (equals < 0) return part;
      removed += part.length - equals - 1;
      return part.slice(0, equals + 1);
    }).join("&");
    url.search = clean;
    return { applied: removed > 0, url: url.href, parameterCount: count,
      originalQueryLength: new URL(rawUrl).search.length - (count ? 1 : 0), removedValueCharacters: removed };
  }

  function hostname(rawUrl) {
    try { return new URL(rawUrl).hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, ""); }
    catch (_) { return ""; }
  }

  function popularity(rawUrl, catalog) {
    const host = hostname(rawUrl);
    if (!host || !catalog?.popular_domain_ranks) return null;
    const domain = typeof catalog.registrableDomainForHost === "function"
      ? catalog.registrableDomainForHost(host) : host;
    const rank = catalog.popular_domain_ranks[domain];
    return rank ? { domain, rank, trusted: false } : null;
  }

  function ipv4Parts(host) {
    const parts = String(host).split(".");
    if (parts.length !== 4 || !parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255)) return null;
    return parts.map(Number);
  }

  function ipAddress(rawUrl) {
    const host = hostname(rawUrl);
    if (!host) return null;
    const parts = ipv4Parts(host);
    if (parts) {
      const [a, b] = parts;
      const local = a === 10 || a === 127 || a === 0 || (a === 169 && b === 254)
        || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
      return { host, version: 4, local };
    }
    if (host.includes(":")) {
      const normalized = host.toLowerCase();
      const local = normalized === "::1" || normalized === "::" || normalized.startsWith("fe8")
        || normalized.startsWith("fe9") || normalized.startsWith("fea") || normalized.startsWith("feb")
        || normalized.startsWith("fc") || normalized.startsWith("fd");
      return { host, version: 6, local };
    }
    return null;
  }

  function isTrusted(rawUrl, trustedHosts) {
    const host = hostname(rawUrl);
    return Boolean(host && trustedHosts.some((value) => String(value).toLowerCase().replace(/\.$/, "") === host));
  }

  function decide(result, trustedHosts = [], options = {}) {
    const modelTier = result?.tier ?? "unknown";
    const mode = options?.mode === "passive" ? "passive" : "normal";
    const continuedOnce = options?.continuedOnce === true;
    if (isTrusted(result?.rawUrl ?? "", trustedHosts)) {
      return {
        modelTier, effectiveTier: "trusted", interrupt: false, mode,
        headline: "Allowed by you",
        explanation: "You added this exact website name to your allowed-sites list. The detector still records its result but will not interrupt the page.",
      };
    }
    const service = servicePage(result?.rawUrl ?? "", options.sitePolicy);
    if (service && userControlledServiceContent(result?.rawUrl ?? "", service)) {
      return sharedContentDecision(result, service, mode, continuedOnce);
    }
    if (service) return {
      modelTier, effectiveTier: "allow", interrupt: false, mode,
      reason: "recognized_service", serviceRule: service.ruleId, policyVersion: POLICY_VERSION,
      popularityRank: service.popularityRank,
      headline: "Recognised service",
      explanation: "This address matches a reviewed official service page, so no URL-text warning is shown. Links and redirect destinations are checked separately.",
    };
    const ip = ipAddress(result?.rawUrl ?? "");
    if (ip) {
      return {
        modelTier, effectiveTier: ip.local ? "ip_local" : "ip_review", interrupt: false, mode, ip,
        headline: ip.local ? "Local network address" : "Public IP address",
        explanation: ip.local
          ? "This looks like a valid address on your local network. It is shown for review but is not blocked using the URL-text score alone."
          : "This is a valid public IP address. Because it has no website name for the detector to examine, it is shown for review rather than blocked automatically.",
      };
    }
    const presentation = {
      allow: ["No warning found", "The detector did not find a strong phishing pattern in this URL."],
      secondary_review: ["Review suggested", "This URL looks unusual, but there is not enough evidence to call it phishing. You may continue."],
      caution: ["Use caution", "Some parts of this URL look suspicious. This does not prove the site is phishing, and the page remains available."],
      warning: ["High-risk URL", "This URL strongly resembles phishing patterns. Navigation is paused, but you can choose to continue."],
      block: ["Very high-risk URL", "This URL reached the highest risk level. Navigation is paused, but you can choose to continue."],
    }[modelTier] ?? ["URL review unavailable", "The model returned an unknown policy tier."];
    const blockingTier = modelTier === "warning" || modelTier === "block";
    const passiveBlock = mode === "passive" && blockingTier;
    const continuedBlock = continuedOnce && blockingTier;
    return {
      modelTier, effectiveTier: modelTier,
      reason: "model_policy", popularity: popularity(result?.rawUrl ?? "", options.sitePolicy),
      interrupt: mode === "normal" && blockingTier && !continuedBlock,
      mode, continuedOnce: continuedBlock,
      headline: continuedBlock
        ? `${modelTier === "block" ? "Very high-risk" : "High-risk"} pattern — continued once`
        : passiveBlock ? `${modelTier === "block" ? "Very high-risk" : "High-risk"} pattern — passive mode` : presentation[0],
      explanation: continuedBlock
        ? "You chose to continue once. The warning remains active, so avoid entering passwords, payment details, or other sensitive information."
        : passiveBlock
        ? "The URL reached the highest risk level, but Passive mode shows the warning without interrupting navigation."
        : presentation[1],
    };
  }

  function shouldShowAdvisory(decision) {
    if (!decision || decision.interrupt !== false) return false;
    return ["secondary_review", "caution", "warning", "block", "ip_review"]
      .includes(decision.effectiveTier);
  }

  return { POLICY_VERSION, servicePage, userControlledServiceContent, googleDocumentQueryNoise,
    googleCollaborativeDocumentRoute, sharedContentDecision,
    popularity, queryView, hostname, ipAddress, isTrusted, decide, shouldShowAdvisory };
});
