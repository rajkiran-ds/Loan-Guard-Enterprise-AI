"""
src/agent/graph.py
------------------
Agentic workflow for loan underwriting using LangGraph.
"""

import inspect
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.ml.model import RiskModel
from src.decision.rule_engine import evaluate, ApplicantSnapshot

class LoanState(TypedDict):
    application: dict
    ml_probability: float
    risk_level: str
    rule_decision: str
    rule_triggered: str
    requires_review: bool
    final_status: str

def node_ml_score(state: LoanState):
    """Runs the PyTorch MLP risk model."""
    features = state["application"]
    assessment = RiskModel.predict(features)
    return {
        "ml_probability": assessment.default_probability,
        "risk_level": assessment.risk_category
    }

def node_evaluate_rules(state: LoanState):
    """Runs the deterministic rule engine."""
    # 1. Start with the raw application data
    app_data = state["application"].copy()
    
    # 2. Map annual to monthly for the rule engine
    if "annual_income" in app_data and "monthly_income" not in app_data:
        app_data["monthly_income"] = app_data["annual_income"] / 12.0

    # 3. Inject the ML model outputs into the data!
    app_data["risk_category"] = state.get("risk_level")
    app_data["default_probability"] = state.get("ml_probability")

    # 4. Dynamically filter to ONLY include fields ApplicantSnapshot accepts
    sig = inspect.signature(ApplicantSnapshot)
    valid_keys = sig.parameters.keys()
    filtered_data = {k: v for k, v in app_data.items() if k in valid_keys}
    
    # 5. Check for any remaining missing fields before crashing
    missing = [k for k in valid_keys if k not in filtered_data and sig.parameters[k].default == inspect.Parameter.empty]
    if missing:
        raise ValueError(f"Missing required fields for rule engine: {missing}. Please add them to the application data.")

    applicant = ApplicantSnapshot(**filtered_data)
    decision = evaluate(applicant)
    
    return {
        "rule_decision": decision.outcome,
        "rule_triggered": decision.rule_triggered,
        "requires_review": decision.requires_manual_review
    }

def node_human_review(state: LoanState):
    return {"final_status": "Pending Manual Review"}

def node_finalize(state: LoanState):
    if state.get("requires_review") and state.get("final_status") == "Pending Manual Review":
         status = state["rule_decision"] 
    elif state.get("requires_review"):
        status = state.get("final_status")
    else:
        status = state["rule_decision"]
    return {"final_status": status}

def route_after_rules(state: LoanState) -> Literal["human_review", "finalize"]:
    if state["requires_review"]:
        return "human_review"
    return "finalize"

# --- Build Graph ---
builder = StateGraph(LoanState)

builder.add_node("score_ml", node_ml_score)
builder.add_node("evaluate_rules", node_evaluate_rules)
builder.add_node("human_review", node_human_review)
builder.add_node("finalize", node_finalize)

builder.set_entry_point("score_ml")
builder.add_edge("score_ml", "evaluate_rules")
builder.add_conditional_edges("evaluate_rules", route_after_rules)
builder.add_edge("human_review", "finalize")
builder.add_edge("finalize", END)

memory = MemorySaver()
loan_agent = builder.compile(checkpointer=memory, interrupt_before=["human_review"])