# Component Specifications Index

DB1 Subordinate Requirements Specifications Pack
================================================

0. Document Conventions

-----------------------

### 0.1 Requirement ID scheme

* Format: `<SUBSYSTEM>-<TYPE>-<NNN>`
  
  * `TYPE`: `DATA` (data contract), `IF` (interface), `FR` (functional), `NFR` (non-functional), `TEST` (verification)

* Priority tags:
  
  * **P0** = required for DB1
  
  * **P1** = planned for DB2/Phase 2
  
  * **P2** = Phase 3+

* Every requirement includes:
  
  * **Statement** (single “shall”)
  
  * **Rationale**
  
  * **Acceptance criteria**

### 0.2 Subsystem codes

`EVT, HOOK, CFG, SEC, PERSIST, OBS, SB, ORCH, ING, FEAT, SIG, RISK, EXEC, PORT, LEDG, ANA, OPT, GOV, KILL, CB, HEL, SYN, CTX, CG, LEARN, BENCH`

### 0.3 System planes

* **Execution Plane**: market → signal → risk → order → fills → portfolio → analytics

* **Cognitive Plane**: Synapse (retrieve) → Cortex (reason) → Helix (model I/O) + CodeGen (patch ops)

* **Control Plane**: Governance + Observability + Config + Security + Persistence

Generated specifications for all previously defined L2 and L3 components.

## L2 Components

| L2 ID   | Component                                  | Spec                                                                                                          | Children (L3)                                                                |
| ------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `L2-01` | EVT — Event Model & Signal Set             | [L2-01_EVT_Event_Model_and_Signal_Set.md](L2/L2-01_EVT_Event_Model_and_Signal_Set.md)                         | [L3-01](L3/L3-01_EVT-SIGNALS_Event_Ontology_and_Schemas.md)                  |
| `L2-02` | HOOK — Hook & Middleware Framework         | [L2-02_HOOK_Hook_and_Middleware_Framework.md](L2/L2-02_HOOK_Hook_and_Middleware_Framework.md)                 | [L3-03](L3/L3-03_HOOK-PIPELINE_Hook_Pipeline_and_Context.md)                 |
| `L2-03` | CFG — Configuration & Profiles             | [L2-03_CFG_Configuration_and_Profiles.md](L2/L2-03_CFG_Configuration_and_Profiles.md)                         | _None_                                                                       |
| `L2-04` | SEC — Security, Identity & Secrets         | [L2-04_SEC_Security_Identity_and_Secrets.md](L2/L2-04_SEC_Security_Identity_and_Secrets.md)                   | [L3-15](L3/L3-15_SEC-SECRETS_Secrets_and_Key_Management.md)                  |
| `L2-05` | PERSIST — Persistence Layer                | [L2-05_PERSIST_Persistence_Layer.md](L2/L2-05_PERSIST_Persistence_Layer.md)                                   | _None_                                                                       |
| `L2-06` | OBS — Observability                        | [L2-06_OBS_Observability.md](L2/L2-06_OBS_Observability.md)                                                   | [L3-14](L3/L3-14_OBS-SLOS_SLOs_Alerts_and_Dashboards.md)                     |
| `L2-07` | SB — StreamingBus                          | [L2-07_SB_StreamingBus.md](L2/L2-07_SB_StreamingBus.md)                                                       | [L3-02](L3/L3-02_SB-ADAPTERS_StreamingBus_Adapters.md)                       |
| `L2-08` | ORCH — OrchestrationEngine                 | [L2-08_ORCH_OrchestrationEngine.md](L2/L2-08_ORCH_OrchestrationEngine.md)                                     | _None_                                                                       |
| `L2-09` | ING — Ingestion & Normalization            | [L2-09_ING_Ingestion_and_Normalization.md](L2/L2-09_ING_Ingestion_and_Normalization.md)                       | _None_                                                                       |
| `L2-10` | FEAT — Feature Engineering & Data Mining   | [L2-10_FEAT_Feature_Engineering_and_Data_Mining.md](L2/L2-10_FEAT_Feature_Engineering_and_Data_Mining.md)     | _None_                                                                       |
| `L2-11` | SIG — SignalEngine (SignalCore)            | [L2-11_SIG_SignalEngine_SignalCore.md](L2/L2-11_SIG_SignalEngine_SignalCore.md)                               | [L3-06](L3/L3-06_SIG-STRATEGY-PACKS_Strategy_Pack_API_and_Model_Runner.md)   |
| `L2-12` | RISK — RiskEngine (RiskGuard)              | [L2-12_RISK_RiskEngine_RiskGuard.md](L2/L2-12_RISK_RiskEngine_RiskGuard.md)                                   | [L3-05](L3/L3-05_RISK-POLICY-PACKS_Risk_Policy_Packs_and_Reason_Codes.md)    |
| `L2-13` | EXEC — ExecutionEngine (FlowRoute)         | [L2-13_EXEC_ExecutionEngine_FlowRoute.md](L2/L2-13_EXEC_ExecutionEngine_FlowRoute.md)                         | [L3-04](L3/L3-04_EXEC-BROKER-ADAPTER_Broker_Adapter_Contract.md)             |
| `L2-14` | PORT — PortfolioEngine                     | [L2-14_PORT_PortfolioEngine.md](L2/L2-14_PORT_PortfolioEngine.md)                                             | _None_                                                                       |
| `L2-15` | LEDG — LedgerEngine                        | [L2-15_LEDG_LedgerEngine.md](L2/L2-15_LEDG_LedgerEngine.md)                                                   | [L3-07](L3/L3-07_LEDG-ACCOUNTING_Ledger_Accounting_Rules_and_Invariants.md)  |
| `L2-16` | ANA — AnalyticsEngine (ProofBench)         | [L2-16_ANA_AnalyticsEngine_ProofBench.md](L2/L2-16_ANA_AnalyticsEngine_ProofBench.md)                         | _None_                                                                       |
| `L2-17` | OPT — PortfolioOptEngine (QPO wrapper)     | [L2-17_OPT_PortfolioOptEngine_QPO_Wrapper.md](L2/L2-17_OPT_PortfolioOptEngine_QPO_Wrapper.md)                 | _None_                                                                       |
| `L2-18` | GOV — GovernanceEngine                     | [L2-18_GOV_GovernanceEngine.md](L2/L2-18_GOV_GovernanceEngine.md)                                             | _None_                                                                       |
| `L2-19` | KILL — KillSwitch                          | [L2-19_KILL_KillSwitch.md](L2/L2-19_KILL_KillSwitch.md)                                                       | _None_                                                                       |
| `L2-20` | CB — CircuitBreaker                        | [L2-20_CB_CircuitBreaker.md](L2/L2-20_CB_CircuitBreaker.md)                                                   | _None_                                                                       |
| `L2-21` | HEL — Helix LLM Provider                   | [L2-21_HEL_Helix_LLM_Provider.md](L2/L2-21_HEL_Helix_LLM_Provider.md)                                         | [L3-08](L3/L3-08_HEL-MODEL-ROUTER_Model_Routing_and_Allowlisting.md)         |
| `L2-22` | SYN — Synapse RAG Engine                   | [L2-22_SYN_Synapse_RAG_Engine.md](L2/L2-22_SYN_Synapse_RAG_Engine.md)                                         | [L3-09](L3/L3-09_SYN-INDEX-RETRIEVE_Indexing_and_Retrieval_Pipeline.md)      |
| `L2-23` | CTX — Cortex LLM Engine                    | [L2-23_CTX_Cortex_LLM_Engine.md](L2/L2-23_CTX_Cortex_LLM_Engine.md)                                           | [L3-10](L3/L3-10_CTX-WORKFLOWS_Cortex_Workflows_and_Output_Schemas.md)       |
| `L2-24` | CG — CodeGenService                        | [L2-24_CG_CodeGenService.md](L2/L2-24_CG_CodeGenService.md)                                                   | [L3-11](L3/L3-11_CG-PATCH-LIFECYCLE_Patch_Lifecycle_and_Guardrails.md)       |
| `L2-25` | LEARN — LearningEngine                     | [L2-25_LEARN_LearningEngine.md](L2/L2-25_LEARN_LearningEngine.md)                                             | [L3-12](L3/L3-12_LEARN-ROLL-OUT_Model_Rollout_and_Drift_Monitoring.md)       |
| `L2-26` | BENCH — Benchmark Packs & Backtest Harness | [L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md](L2/L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md) | [L3-13](L3/L3-13_BENCH-PACKS_Benchmark_Pack_Generator_and_Regime_Tagging.md) |

## L3 Components

| L3 ID   | Component                                               | Spec                                                                                                                                    | Parent (L2)                                                     |
| ------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `L3-01` | EVT-SIGNALS — Event Ontology & Schemas                  | [L3-01_EVT-SIGNALS_Event_Ontology_and_Schemas.md](L3/L3-01_EVT-SIGNALS_Event_Ontology_and_Schemas.md)                                   | [L2-01](L2/L2-01_EVT_Event_Model_and_Signal_Set.md)             |
| `L3-02` | SB-ADAPTERS — StreamingBus Adapters                     | [L3-02_SB-ADAPTERS_StreamingBus_Adapters.md](L3/L3-02_SB-ADAPTERS_StreamingBus_Adapters.md)                                             | [L2-07](L2/L2-07_SB_StreamingBus.md)                            |
| `L3-03` | HOOK-PIPELINE — Hook Pipeline & Context                 | [L3-03_HOOK-PIPELINE_Hook_Pipeline_and_Context.md](L3/L3-03_HOOK-PIPELINE_Hook_Pipeline_and_Context.md)                                 | [L2-02](L2/L2-02_HOOK_Hook_and_Middleware_Framework.md)         |
| `L3-04` | EXEC-BROKER-ADAPTER — Broker Adapter Contract           | [L3-04_EXEC-BROKER-ADAPTER_Broker_Adapter_Contract.md](L3/L3-04_EXEC-BROKER-ADAPTER_Broker_Adapter_Contract.md)                         | [L2-13](L2/L2-13_EXEC_ExecutionEngine_FlowRoute.md)             |
| `L3-05` | RISK-POLICY-PACKS — Risk Policy Packs & Reason Codes    | [L3-05_RISK-POLICY-PACKS_Risk_Policy_Packs_and_Reason_Codes.md](L3/L3-05_RISK-POLICY-PACKS_Risk_Policy_Packs_and_Reason_Codes.md)       | [L2-12](L2/L2-12_RISK_RiskEngine_RiskGuard.md)                  |
| `L3-06` | SIG-STRATEGY-PACKS — Strategy Pack API & Model Runner   | [L3-06_SIG-STRATEGY-PACKS_Strategy_Pack_API_and_Model_Runner.md](L3/L3-06_SIG-STRATEGY-PACKS_Strategy_Pack_API_and_Model_Runner.md)     | [L2-11](L2/L2-11_SIG_SignalEngine_SignalCore.md)                |
| `L3-07` | LEDG-ACCOUNTING — Ledger Accounting Rules & Invariants  | [L3-07_LEDG-ACCOUNTING_Ledger_Accounting_Rules_and_Invariants.md](L3/L3-07_LEDG-ACCOUNTING_Ledger_Accounting_Rules_and_Invariants.md)   | [L2-15](L2/L2-15_LEDG_LedgerEngine.md)                          |
| `L3-08` | HEL-MODEL-ROUTER — Model Routing & Allowlisting         | [L3-08_HEL-MODEL-ROUTER_Model_Routing_and_Allowlisting.md](L3/L3-08_HEL-MODEL-ROUTER_Model_Routing_and_Allowlisting.md)                 | [L2-21](L2/L2-21_HEL_Helix_LLM_Provider.md)                     |
| `L3-09` | SYN-INDEX-RETRIEVE — Indexing & Retrieval Pipeline      | [L3-09_SYN-INDEX-RETRIEVE_Indexing_and_Retrieval_Pipeline.md](L3/L3-09_SYN-INDEX-RETRIEVE_Indexing_and_Retrieval_Pipeline.md)           | [L2-22](L2/L2-22_SYN_Synapse_RAG_Engine.md)                     |
| `L3-10` | CTX-WORKFLOWS — Cortex Workflows & Output Schemas       | [L3-10_CTX-WORKFLOWS_Cortex_Workflows_and_Output_Schemas.md](L3/L3-10_CTX-WORKFLOWS_Cortex_Workflows_and_Output_Schemas.md)             | [L2-23](L2/L2-23_CTX_Cortex_LLM_Engine.md)                      |
| `L3-11` | CG-PATCH-LIFECYCLE — Patch Lifecycle & Guardrails       | [L3-11_CG-PATCH-LIFECYCLE_Patch_Lifecycle_and_Guardrails.md](L3/L3-11_CG-PATCH-LIFECYCLE_Patch_Lifecycle_and_Guardrails.md)             | [L2-24](L2/L2-24_CG_CodeGenService.md)                          |
| `L3-12` | LEARN-ROLL-OUT — Model Rollout & Drift Monitoring       | [L3-12_LEARN-ROLL-OUT_Model_Rollout_and_Drift_Monitoring.md](L3/L3-12_LEARN-ROLL-OUT_Model_Rollout_and_Drift_Monitoring.md)             | [L2-25](L2/L2-25_LEARN_LearningEngine.md)                       |
| `L3-13` | BENCH-PACKS — Benchmark Pack Generator & Regime Tagging | [L3-13_BENCH-PACKS_Benchmark_Pack_Generator_and_Regime_Tagging.md](L3/L3-13_BENCH-PACKS_Benchmark_Pack_Generator_and_Regime_Tagging.md) | [L2-26](L2/L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md) |
| `L3-14` | OBS-SLOS — SLOs, Alerts & Dashboards                    | [L3-14_OBS-SLOS_SLOs_Alerts_and_Dashboards.md](L3/L3-14_OBS-SLOS_SLOs_Alerts_and_Dashboards.md)                                         | [L2-06](L2/L2-06_OBS_Observability.md)                          |
| `L3-15` | SEC-SECRETS — Secrets & Key Management                  | [L3-15_SEC-SECRETS_Secrets_and_Key_Management.md](L3/L3-15_SEC-SECRETS_Secrets_and_Key_Management.md)                                   | [L2-04](L2/L2-04_SEC_Security_Identity_and_Secrets.md)          |
