from src.agent.graph import loan_agent

config = {"configurable": {"thread_id": "app_123"}}

# Added existing_emis and late_payments
application_data = {
    "cibil_score": 680,         # Between 650 and 720 -> Rule 3 Conditional
    "annual_income": 800000,
    "employment_years": 3,
    "loan_amount": 300000,
    "loan_term_months": 36,
    "residential_assets": 0,
    "commercial_assets": 0,
    "existing_liabilities": 0,
    "existing_emis": 5000,      # <-- NEW
    "late_payments": 0          # <-- NEW
}

print("--- Running Agent ---")
state = loan_agent.invoke({"application": application_data}, config)
print(f"Current State: {state['rule_decision']} (Requires Review: {state['requires_review']})")

if state["requires_review"]:
    print("\n--- Graph Paused for Human Review ---")
    print("Underwriter manually approves the loan.")
    
    loan_agent.update_state(
        config, 
        {"final_status": "Approved by Underwriter (Manual Override)"}
    )
    
    final_state = loan_agent.invoke(None, config)
    print(f"\nFinal Decision: {final_state['final_status']}")
else:
    print(f"\nFinal Decision: {state['final_status']}")