#!/usr/bin/env python3
"""Validate RAG database implementations from SYNAPSE_RAG_DATABASE_REVIEW.md."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def validate_schema():
    """Validate schema.py changes."""
    print("\n=== Validating Schema (R1, R7, R8) ===")
    try:
        from ordinis.adapters.storage.schema import (
            SCHEMA_DDL, 
            SCHEMA_VERSION, 
            get_migration_sql,
            INITIAL_SYSTEM_STATE,
        )
        print(f"  Schema Version: {SCHEMA_VERSION}")
        assert SCHEMA_VERSION == 2, "Schema version should be 2"
        
        # Check for new tables
        assert "CREATE TABLE IF NOT EXISTS sessions" in SCHEMA_DDL
        assert "CREATE TABLE IF NOT EXISTS messages" in SCHEMA_DDL
        assert "CREATE TABLE IF NOT EXISTS session_summaries" in SCHEMA_DDL
        assert "CREATE TABLE IF NOT EXISTS chroma_sync_queue" in SCHEMA_DDL
        assert "CREATE TABLE IF NOT EXISTS retention_audit" in SCHEMA_DDL
        print("  ✓ New tables defined (sessions, messages, session_summaries, chroma_sync_queue, retention_audit)")
        
        # Check FTS5
        assert "CREATE VIRTUAL TABLE IF NOT EXISTS trades_fts" in SCHEMA_DDL
        print("  ✓ FTS5 virtual table defined (trades_fts)")
        
        # Check session_id columns
        assert "session_id TEXT" in SCHEMA_DDL
        print("  ✓ session_id columns added to trading tables")
        
        # Check migration
        migration = get_migration_sql(1, 2)
        assert migration is not None
        assert "ALTER TABLE trades ADD COLUMN session_id" in migration
        print("  ✓ Migration v1→v2 defined")
        
        # Check new system state entries
        state_keys = [s[0] for s in INITIAL_SYSTEM_STATE]
        assert "current_session_id" in state_keys
        assert "chroma_sync_enabled" in state_keys
        print("  ✓ New system state entries added")
        
        print("  ✓ schema.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ schema.py validation FAILED: {e}")
        return False

def validate_id_generator():
    """Validate id_generator.py (R5)."""
    print("\n=== Validating ID Generator (R5) ===")
    try:
        from ordinis.rag.vectordb.id_generator import (
            generate_vector_id,
            generate_trade_vector_id,
            generate_session_chunk_id,
            generate_content_hash,
            parse_vector_id,
            is_valid_vector_id,
            VectorIdGenerator,
        )
        
        # Test basic ID generation
        vid = generate_vector_id("trade", "t_123", "AAPL long entry")
        assert vid.startswith("trade:t_123:")
        print(f"  Generated ID: {vid}")
        
        # Test trade ID
        trade_id = generate_trade_vector_id("t_456", "BUY 100 shares")
        assert trade_id.startswith("trade:t_456:")
        
        # Test session chunk ID
        chunk_id = generate_session_chunk_id("sess_001", "chunk content", 0)
        assert chunk_id.startswith("session:sess_001:")
        
        # Test determinism (same input = same output)
        vid1 = generate_vector_id("trade", "t_123", "same content")
        vid2 = generate_vector_id("trade", "t_123", "same content")
        assert vid1 == vid2, "IDs should be deterministic"
        print("  ✓ IDs are deterministic")
        
        # Test collision resistance (different content = different ID)
        vid3 = generate_vector_id("trade", "t_123", "different content")
        assert vid1 != vid3, "Different content should produce different IDs"
        print("  ✓ Content changes produce different IDs")
        
        # Test parsing
        parsed = parse_vector_id(vid)
        assert parsed["entity_type"] == "trade"
        assert parsed["source_id"] == "t_123"
        print("  ✓ ID parsing works")
        
        # Test validation
        assert is_valid_vector_id(vid)
        assert not is_valid_vector_id("invalid-id-format")
        print("  ✓ ID validation works")
        
        # Test factory class
        gen = VectorIdGenerator()
        tid, meta = gen.create_trade_id("t_789", "test content")
        assert "entity_type" in meta
        assert "embedding_model" in meta
        print("  ✓ VectorIdGenerator factory works")
        
        print("  ✓ id_generator.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ id_generator.py validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_dual_write():
    """Validate dual_write.py (R3)."""
    print("\n=== Validating DualWriteManager (R3) ===")
    try:
        from ordinis.core.dual_write import (
            DualWriteManager,
            DualWriteResult,
            WritePhase,
            SyncQueueEntry,
            dual_write,
        )
        
        # Check classes exist and have expected attributes
        assert hasattr(DualWriteManager, "write_trade_with_vector")
        assert hasattr(DualWriteManager, "write_order_with_vector")
        assert hasattr(DualWriteManager, "force_sync_all")
        print("  ✓ DualWriteManager has expected methods")
        
        # Check result dataclass
        result = DualWriteResult(
            success=True,
            transaction_id="test",
            phase=WritePhase.COMPLETED,
        )
        assert result.to_dict()["success"] is True
        print("  ✓ DualWriteResult works")
        
        # Check write phases
        assert WritePhase.PENDING.value == "pending"
        assert WritePhase.COMPLETED.value == "completed"
        assert WritePhase.COMPENSATING.value == "compensating"
        print("  ✓ WritePhase enum correct")
        
        print("  ✓ dual_write.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ dual_write.py validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_trade_ingester():
    """Validate trade_ingester.py (R2)."""
    print("\n=== Validating TradeVectorIngester (R2) ===")
    try:
        from ordinis.rag.pipeline.trade_ingester import (
            TradeVectorIngester,
            SyncMode,
            SyncResult,
        )
        
        # Check classes exist
        assert hasattr(TradeVectorIngester, "sync_unsynced_trades")
        assert hasattr(TradeVectorIngester, "full_reindex")
        assert hasattr(TradeVectorIngester, "get_sync_status")
        print("  ✓ TradeVectorIngester has expected methods")
        
        # Check sync modes
        assert SyncMode.INCREMENTAL.value == "incremental"
        assert SyncMode.FULL.value == "full"
        assert SyncMode.SESSION.value == "session"
        print("  ✓ SyncMode enum correct")
        
        # Check result
        result = SyncResult(mode=SyncMode.INCREMENTAL, synced=10, total_found=15)
        assert result.success_rate < 100
        print("  ✓ SyncResult works")
        
        print("  ✓ trade_ingester.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ trade_ingester.py validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_summarizer():
    """Validate summarizer.py (Memory)."""
    print("\n=== Validating SessionSummarizer (Memory) ===")
    try:
        from ordinis.rag.memory.summarizer import (
            SessionSummarizer,
            Summary,
            SummaryType,
            SummaryStatus,
            SummaryPrompts,
        )
        
        # Check classes exist
        assert hasattr(SessionSummarizer, "create_chunk_summary")
        assert hasattr(SessionSummarizer, "create_session_summary")
        assert hasattr(SessionSummarizer, "search_summaries")
        print("  ✓ SessionSummarizer has expected methods")
        
        # Check summary types
        assert SummaryType.CHUNK.value == "chunk"
        assert SummaryType.SESSION.value == "session"
        print("  ✓ SummaryType enum correct")
        
        # Check prompts
        assert "{messages}" in SummaryPrompts.CHUNK_SUMMARY
        assert "{chunks}" in SummaryPrompts.SESSION_SUMMARY
        print("  ✓ SummaryPrompts defined")
        
        print("  ✓ summarizer.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ summarizer.py validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_context_assembler():
    """Validate assembler.py (Memory)."""
    print("\n=== Validating ContextAssembler (Memory) ===")
    try:
        from ordinis.rag.context.assembler import (
            ContextAssembler,
            ContextSource,
            ContextPriority,
            ContextChunk,
            AssembledContext,
        )
        
        # Check classes exist
        assert hasattr(ContextAssembler, "assemble_context")
        assert hasattr(ContextAssembler, "preload_session_context")
        print("  ✓ ContextAssembler has expected methods")
        
        # Check sources
        assert ContextSource.RECENT_MESSAGES.value == "recent_messages"
        assert ContextSource.TRADE_HISTORY.value == "trade_history"
        print("  ✓ ContextSource enum correct")
        
        # Check priorities
        assert ContextPriority.CRITICAL.value < ContextPriority.LOW.value
        print("  ✓ ContextPriority ordering correct")
        
        print("  ✓ assembler.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ assembler.py validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_retention_manager():
    """Validate retention.py (Memory)."""
    print("\n=== Validating RetentionManager (Memory) ===")
    try:
        from ordinis.rag.memory.retention import (
            RetentionManager,
            RetentionPolicy,
            RetentionAction,
            RetentionResult,
            EntityType,
            DEFAULT_POLICIES,
        )
        
        # Check classes exist
        assert hasattr(RetentionManager, "enforce_retention_policies")
        assert hasattr(RetentionManager, "get_retention_stats")
        assert hasattr(RetentionManager, "get_audit_history")
        print("  ✓ RetentionManager has expected methods")
        
        # Check policies
        assert len(DEFAULT_POLICIES) >= 4
        print(f"  ✓ {len(DEFAULT_POLICIES)} default policies defined")
        
        # Check entity types
        assert EntityType.TRADE.value == "trade"
        assert EntityType.MESSAGE.value == "message"
        print("  ✓ EntityType enum correct")
        
        # Check actions
        assert RetentionAction.ARCHIVE.value == "archive"
        assert RetentionAction.PURGE.value == "purge"
        print("  ✓ RetentionAction enum correct")
        
        print("  ✓ retention.py validation PASSED")
        return True
    except Exception as e:
        print(f"  ✗ retention.py validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validations."""
    print("=" * 60)
    print("SYNAPSE_RAG_DATABASE_REVIEW Implementation Validation")
    print("=" * 60)
    
    results = []
    results.append(("Schema (R1, R7, R8)", validate_schema()))
    results.append(("ID Generator (R5)", validate_id_generator()))
    results.append(("DualWriteManager (R3)", validate_dual_write()))
    results.append(("TradeVectorIngester (R2)", validate_trade_ingester()))
    results.append(("SessionSummarizer", validate_summarizer()))
    results.append(("ContextAssembler", validate_context_assembler()))
    results.append(("RetentionManager", validate_retention_manager()))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 All implementations validated successfully!")
        return 0
    else:
        print("\n⚠️  Some validations failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
