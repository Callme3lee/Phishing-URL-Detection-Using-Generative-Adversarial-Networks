(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.PhishingPilotStore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SCHEMA_VERSION = 6;
  const DECISION_TIERS = [
    "trusted", "allow", "ip_local", "secondary_review",
    "ip_review", "caution", "warning", "block", "unknown",
  ];

  function emptyDecisionCounts() {
    return Object.fromEntries(DECISION_TIERS.map((tier) => [tier, 0]));
  }

  function emptyCounters() {
    return {
      uniqueAutomaticUrls: 0,
      automaticInferenceExecutions: 0,
      duplicateAssessmentsSuppressed: 0,
      manualScans: 0,
      inferenceErrors: 0,
      falseAlarmReports: 0,
      missedPhishingReports: 0,
      warningContinues: 0,
      trustedSiteActions: 0,
      advisoryDismissals: 0,
      feedbackSubmissions: 0,
      feedbackRecordsEvicted: 0,
      navigationInterrupts: 0,
      untrackedAssessments: 0,
      advisoryDisplays: 0,
      servicePageDecisions: 0,
      modelViewEvaluations: 0,
    };
  }

  function emptyEventCounts() {
    return {
      navigation: 0,
      history_state: 0,
      fragment: 0,
      advisory_recovery: 0,
      policy_change: 0,
      unknown: 0,
    };
  }

  function emptyLatency() {
    return { count: 0, totalMilliseconds: 0, minimumMilliseconds: null, maximumMilliseconds: null };
  }

  function createStudy(participantId, now, extensionVersion) {
    return {
      schemaVersion: SCHEMA_VERSION,
      studyName: "Phishing URL Guard local detection activity",
      participantId,
      extensionVersion,
      decisionPolicy: "reviewed_site_catalog_v3",
      createdAt: now,
      consentedAt: now,
      measurementStartedAt: now,
      lastUpdatedAt: now,
      privacy: {
        automaticRawUrlsStored: false,
        localSaltedFingerprintsUsedForDeduplication: true,
        deduplicationFingerprintsExported: false,
        feedbackUrlsStoredOnlyAfterUserAction: true,
        queryAndFragmentRemovedFromFeedbackUrls: true,
        automaticUpload: false,
      },
      measurement: {
        primaryUnit: "unique canonical URL checked while local history is enabled",
        fragmentOnlyChangesCountAsNewUrl: false,
        repeatedCanonicalUrlsCountOnce: true,
        manualScansExcludedFromUniqueUrlDecisions: true,
        modelExecutionsReportedSeparately: true,
      },
      counters: emptyCounters(),
      decisionCounts: emptyDecisionCounts(),
      duplicateEventCounts: emptyEventCounts(),
      latency: emptyLatency(),
      feedbackReports: [],
      assessmentRecords: [],
      interventionRecords: [],
    };
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function migrateStudy(study, now, extensionVersion) {
    if (!study) return { study: null, migrated: false };
    if (Number(study.schemaVersion) >= SCHEMA_VERSION) {
      study.extensionVersion = extensionVersion ?? study.extensionVersion;
      study.counters = { ...emptyCounters(), ...(study.counters ?? {}) };
      study.decisionCounts = { ...emptyDecisionCounts(), ...(study.decisionCounts ?? {}) };
      study.duplicateEventCounts = { ...emptyEventCounts(), ...(study.duplicateEventCounts ?? {}) };
      study.latency = { ...emptyLatency(), ...(study.latency ?? {}) };
      study.feedbackReports = Array.isArray(study.feedbackReports) ? study.feedbackReports : [];
      study.assessmentRecords = Array.isArray(study.assessmentRecords) ? study.assessmentRecords : [];
      study.interventionRecords = Array.isArray(study.interventionRecords) ? study.interventionRecords : [];
      return { study, migrated: false };
    }

    const legacy = clone(study);
    const migrated = createStudy(
      study.participantId,
      now,
      extensionVersion ?? study.extensionVersion ?? "0.5.1",
    );
    migrated.createdAt = study.createdAt ?? now;
    migrated.consentedAt = study.consentedAt ?? study.createdAt ?? now;
    // Start a new measurement period: legacy feedback cannot be joined reliably
    // to automatic URL identities. Preserve the entire previous export separately.
    migrated.legacyStudy = legacy;
    migrated.feedbackReports = [];
    migrated.counters.falseAlarmReports = migrated.feedbackReports.filter(
      (entry) => entry.claim === "false_alarm",
    ).length;
    migrated.counters.missedPhishingReports = migrated.feedbackReports.filter(
      (entry) => entry.claim === "missed_phishing",
    ).length;
    migrated.legacyV05Telemetry = {
      schemaVersion: Number(legacy.schemaVersion) || 1,
      extensionVersion: legacy.extensionVersion ?? "0.5.0",
      createdAt: legacy.createdAt ?? null,
      lastUpdatedAt: legacy.lastUpdatedAt ?? null,
      counters: clone(legacy.counters ?? {}),
      decisionCounts: clone(legacy.decisionCounts ?? {}),
      latency: clone(legacy.latency ?? emptyLatency()),
      reliabilityNotice: "Archived results use an earlier measurement or decision policy and are excluded from this period.",
    };
    migrated.migration = {
      migratedAt: now,
      fromSchemaVersion: Number(legacy.schemaVersion) || 1,
      reason: "Start local detection activity for v0.5.9; preserve earlier results as legacyStudy.",
    };
    return { study: migrated, migrated: true };
  }

  function canonicalAssessmentIdentity(result) {
    const raw = String(result?.canonicalUrl ?? result?.rawUrl ?? "").trim();
    if (!raw) throw new Error("A checked URL is required for repeat-check handling.");
    const withoutFragment = raw.split("#", 1)[0];
    return withoutFragment || raw;
  }

  function recordLatency(study, result) {
    const latency = Number(result?.inferenceMilliseconds);
    if (!Number.isFinite(latency) || latency < 0) return;
    study.latency.count += 1;
    study.latency.totalMilliseconds += latency;
    study.latency.minimumMilliseconds = study.latency.minimumMilliseconds === null
      ? latency : Math.min(study.latency.minimumMilliseconds, latency);
    study.latency.maximumMilliseconds = study.latency.maximumMilliseconds === null
      ? latency : Math.max(study.latency.maximumMilliseconds, latency);
  }

  function recordInference(study, result, source, now) {
    study.counters.modelViewEvaluations += result.modelViewCount ?? 1;
    if (result.modelError) study.counters.inferenceErrors += 1;
    if (source === "manual") study.counters.manualScans += 1;
    else study.counters.automaticInferenceExecutions += 1;
    recordLatency(study, result);
    study.lastUpdatedAt = now;
    return study;
  }

  function recordAutomaticAssessment(study, result, isUnique, eventSource, now) {
    const source = eventSource in study.duplicateEventCounts ? eventSource : "unknown";
    if (isUnique) {
      study.counters.uniqueAutomaticUrls += 1;
      if (result.decision?.reason === "recognized_service") study.counters.servicePageDecisions += 1;
      const tier = result?.decision?.effectiveTier ?? result?.tier ?? "unknown";
      study.decisionCounts[tier in study.decisionCounts ? tier : "unknown"] += 1;
      study.assessmentRecords.push({
        id: result.assessmentId, tier,
        modelTier: result.tier, decisionReason: result.decision?.reason ?? "model_policy",
        serviceRule: result.decision?.serviceRule ?? null,
        explanationSignals: (result.explanation?.signals ?? []).map((signal) => signal.code).slice(0, 4),
        mode: result.decision?.mode ?? "normal", assessedAt: now,
      });
    } else {
      study.counters.duplicateAssessmentsSuppressed += 1;
      study.duplicateEventCounts[source] += 1;
    }
    study.lastUpdatedAt = now;
    return study;
  }

  function recordError(study, now) {
    study.counters.inferenceErrors += 1;
    study.lastUpdatedAt = now;
    return study;
  }

  function recordAction(study, action, now) {
    const counters = {
      warning_continue: "warningContinues",
      trusted_site: "trustedSiteActions",
      advisory_dismiss: "advisoryDismissals",
    };
    const counter = counters[action];
    if (counter) study.counters[counter] += 1;
    study.lastUpdatedAt = now;
    return study;
  }

  function sanitizeFeedbackUrl(rawUrl) {
    const parsed = new URL(rawUrl);
    if (!/^https?:$/.test(parsed.protocol)) throw new Error("Only HTTP(S) URLs can be reported.");
    const queryParameterCount = [...parsed.searchParams.keys()].length;
    const hadFragment = Boolean(parsed.hash);
    parsed.username = "";
    parsed.password = "";
    parsed.search = "";
    parsed.hash = "";
    return {
      url: parsed.toString(),
      queryValuesRemoved: queryParameterCount > 0,
      queryParameterCount,
      fragmentRemoved: hadFragment,
    };
  }

  function addFeedback(study, input, now, id) {
    if (!["false_alarm", "missed_phishing"].includes(input.claim)) throw new Error("Unsupported feedback type.");
    const sanitized = sanitizeFeedbackUrl(input.url);
    const report = {
      id,
      reportedAt: now,
      claim: input.claim,
      reportedUrl: sanitized.url,
      queryValuesRemoved: sanitized.queryValuesRemoved,
      queryParameterCount: sanitized.queryParameterCount,
      fragmentRemoved: sanitized.fragmentRemoved,
      score: Number.isFinite(input.score) ? input.score : null,
      modelTier: input.modelTier ?? null,
      effectiveTier: input.effectiveTier ?? null,
      identityId: input.identityId ?? null,
      assessmentId: input.assessmentId ?? null,
      source: input.source ?? "legacy",
      mode: input.mode ?? null,
      decisionReason: input.decisionReason ?? "model_policy",
      explanationSignals: Array.isArray(input.explanationSignals)
        ? [...new Set(input.explanationSignals.map(String))].slice(0, 4) : [],
      queryView: input.queryView ? { applied: Boolean(input.queryView.applied),
        score: Number.isFinite(input.queryView.score) ? input.queryView.score : null,
        tier: input.queryView.tier ?? null, parameterCount: input.queryView.parameterCount ?? null,
        originalQueryLength: input.queryView.originalQueryLength ?? null } : null,
    };
    study.counters.feedbackSubmissions += 1;
    study.feedbackReports = study.feedbackReports.filter((entry) => input.identityId
      ? entry.identityId !== input.identityId : entry.reportedUrl !== report.reportedUrl);
    study.feedbackReports.push(report);
    study.counters.feedbackRecordsEvicted += Math.max(0, study.feedbackReports.length - 250);
    study.feedbackReports = study.feedbackReports.slice(-250);
    study.counters.falseAlarmReports = study.feedbackReports.filter((entry) => entry.claim === "false_alarm").length;
    study.counters.missedPhishingReports = study.feedbackReports.filter((entry) => entry.claim === "missed_phishing").length;
    study.lastUpdatedAt = now;
    return report;
  }

  function summary(study) {
    if (!study) return null;
    const counters = { ...emptyCounters(), ...(study.counters ?? {}) };
    return {
      participantId: study.participantId,
      extensionVersion: study.extensionVersion,
      schemaVersion: study.schemaVersion,
      createdAt: study.createdAt,
      measurementStartedAt: study.measurementStartedAt,
      lastUpdatedAt: study.lastUpdatedAt,
      uniqueAutomaticUrls: counters.uniqueAutomaticUrls,
      automaticInferenceExecutions: counters.automaticInferenceExecutions,
      manualScans: counters.manualScans,
      totalModelExecutions: counters.automaticInferenceExecutions + counters.manualScans,
      duplicateAssessmentsSuppressed: counters.duplicateAssessmentsSuppressed,
      feedbackReports: study.feedbackReports.length,
      falseAlarmReports: counters.falseAlarmReports,
      feedbackSubmissions: counters.feedbackSubmissions,
      feedbackRecordsEvicted: counters.feedbackRecordsEvicted,
      navigationInterrupts: counters.navigationInterrupts,
      untrackedAssessments: counters.untrackedAssessments,
      advisoryDisplays: counters.advisoryDisplays,
      servicePageDecisions: counters.servicePageDecisions,
      modelViewEvaluations: counters.modelViewEvaluations,
      missedPhishingReports: counters.missedPhishingReports,
      inferenceErrors: counters.inferenceErrors,
      averageInferenceMilliseconds: study.latency.count
        ? study.latency.totalMilliseconds / study.latency.count : null,
      decisionCounts: { ...emptyDecisionCounts(), ...(study.decisionCounts ?? {}) },
      duplicateEventCounts: { ...emptyEventCounts(), ...(study.duplicateEventCounts ?? {}) },
      legacyTelemetryPresent: Boolean(study.legacyStudy || study.legacyV05Telemetry),
    };
  }

  function summaryCsv(study) {
    const values = summary(study);
    if (!values) return "metric,value\n";
    const rows = [
      ["participant_id", values.participantId],
      ["extension_version", values.extensionVersion],
      ["schema_version", values.schemaVersion],
      ["study_created_at", values.createdAt],
      ["corrected_measurement_started_at", values.measurementStartedAt],
      ["last_updated_at", values.lastUpdatedAt],
      ["unique_automatic_urls", values.uniqueAutomaticUrls],
      ["automatic_inference_executions", values.automaticInferenceExecutions],
      ["manual_scans", values.manualScans],
      ["total_model_executions", values.totalModelExecutions],
      ["duplicate_assessments_suppressed", values.duplicateAssessmentsSuppressed],
      ["false_alarm_reports", values.falseAlarmReports],
      ["feedback_submissions", values.feedbackSubmissions],
      ["feedback_records_evicted", values.feedbackRecordsEvicted],
      ["navigation_interrupts", values.navigationInterrupts],
      ["untracked_assessments", values.untrackedAssessments],
      ["urls_with_displayed_advisory", values.advisoryDisplays],
      ["recognized_service_decisions", values.servicePageDecisions],
      ["model_view_evaluations", values.modelViewEvaluations],
      ["missed_phishing_reports", values.missedPhishingReports],
      ["inference_errors", values.inferenceErrors],
      ["average_inference_ms", values.averageInferenceMilliseconds ?? ""],
      ["legacy_v0_5_telemetry_archived", values.legacyTelemetryPresent],
      ...Object.entries(values.decisionCounts).map(([tier, count]) => [`unique_decision_${tier}`, count]),
      ...Object.entries(values.duplicateEventCounts).map(([event, count]) => [`duplicate_event_${event}`, count]),
    ];
    const escape = (value) => `"${String(value).replaceAll('"', '""')}"`;
    return ["metric,value", ...rows.map(([key, value]) => `${escape(key)},${escape(value)}`)].join("\n") + "\n";
  }

  return {
    SCHEMA_VERSION,
    DECISION_TIERS,
    createStudy,
    migrateStudy,
    canonicalAssessmentIdentity,
    sanitizeFeedbackUrl,
    recordInference,
    recordAutomaticAssessment,
    recordError,
    recordAction,
    addFeedback,
    summary,
    summaryCsv,
  };
});
