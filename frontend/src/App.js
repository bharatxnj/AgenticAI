import React, { useState } from 'react';

const AgenticDashboard = () => {
  const [query, setQuery] = useState('');
  const [steps, setSteps] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Mock function to simulate LangGraph/Multi-Agent Orchestration
  const runAgenticWorkflow = async () => {
    if (!query) return; 
    setIsProcessing(true);
    setSteps([]);
    
    const workflow = [
      { agent: 'Planner', status: 'Analyzing HSBC GCP Architecture...', delay: 1000 },
      { agent: 'Researcher', status: 'Querying BigQuery for MLOps metrics...', delay: 1500 },
      { agent: 'Architect', status: 'Generating Terraform State recommendations...', delay: 1200 },
      { agent: 'Finalizer', status: 'Optimizing UI State Management...', delay: 800 }
    ];

    for (const step of workflow) {
      await new Promise(res => setTimeout(res, step.delay));
      setSteps(prev => [...prev, step]);
    }
    setIsProcessing(false);
  };

  // Inline styles for zero-dependency visibility
  const styles = {
    container: { padding: '40px', backgroundColor: '#111827', minHeight: '100vh', color: 'white', fontFamily: 'sans-serif' },
    header: { color: '#60a5fa', fontSize: '28px', fontWeight: 'bold', marginBottom: '24px' },
    card: { backgroundColor: '#1f2937', padding: '24px', borderRadius: '8px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', border: '1px solid #374151' },
    input: { width: '100%', padding: '16px', borderRadius: '4px', backgroundColor: '#374151', color: 'white', border: '1px solid #4b5563', marginBottom: '16px', fontSize: '16px', boxSizing: 'border-box' },
    button: (processing) => ({
      padding: '12px 24px',
      backgroundColor: processing ? '#4b5563' : '#2563eb',
      color: 'white',
      fontWeight: 'bold',
      borderRadius: '4px',
      border: 'none',
      cursor: processing ? 'not-allowed' : 'pointer',
      transition: 'background-color 0.3s'
    }),
    stepItem: { display: 'flex', alignItems: 'center', padding: '16px', backgroundColor: '#1f2937', borderRadius: '4px', borderLeft: '4px solid #10b981', marginBottom: '16px' },
    agentTag: { color: '#93c5fd', fontFamily: 'monospace', width: '120px', fontWeight: 'bold' },
    placeholder: { color: '#6b7280', fontStyle: 'italic' }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>HSBC Agentic Intelligence Portal</h1>
      
      <div style={styles.card}>
        <input 
          style={styles.input}
          placeholder="Enter project requirement (e.g., 'Analyze GCP VPC Security')..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button 
          onClick={runAgenticWorkflow}
          disabled={isProcessing}
          style={styles.button(isProcessing)}
        >
          {isProcessing ? 'Agent Orchestrating...' : 'Trigger Agentic Workflow'}
        </button>
      </div>

      <div style={{ marginTop: '32px' }}>
        {steps.map((s, i) => (
          <div key={i} style={styles.stepItem}>
            <span style={styles.agentTag}>[{s.agent}]</span>
            <span style={{ marginLeft: '16px' }}>{s.status}</span>
          </div>
        ))}
        {steps.length === 0 && !isProcessing && (
          <p style={styles.placeholder}>Enter a query above to begin the multi-agent analysis.</p>
        )}
      </div>
    </div>
  );
};

export default AgenticDashboard;