import json
import base64
import os
import pandas as pd

# Load dashboard data
with open("dashboard_data.json", "r") as f:
    data = json.load(f)

# Load CSV sample for table (first 200 rows for fast browser rendering with full stats)
df = pd.read_csv("customer_segments_output.csv")
table_records = df[["ID", "Age", "Education", "Marital_Status", "Income", "Total_Spending", "Total_Purchases", "NumWebVisitsMonth", "Segment", "Response"]].head(250).to_dict(orient="records")

# Read generated PNGs as base64 so they render directly in single-file HTML without file path issues
def get_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as img:
            return "data:image/png;base64," + base64.b64encode(img.read()).decode("utf-8")
    return ""

b64_silhouette = get_b64("kmeans_cluster_selection.png")
b64_pca = get_b64("customer_segmentation_pca.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Customer Segmentation & Clustering Output</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #10b981;
      --accent-alt: #8b5cf6;
      --danger: #ef4444;
      --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text-main);
      font-family: var(--font-family);
      line-height: 1.6;
      padding-bottom: 60px;
    }}
    header {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 24px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .badge {{
      background: rgba(59, 130, 246, 0.15);
      border: 1px solid rgba(59, 130, 246, 0.35);
      color: #60a5fa;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .status-dot {{
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
    }}
    .container {{
      max-width: 1350px;
      margin: 0 auto;
      padding: 28px 24px;
    }}
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px 24px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }}
    .kpi-card::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; width: 4px; height: 100%;
      background: var(--primary);
    }}
    .kpi-card.accent::before {{ background: var(--accent); }}
    .kpi-card.purple::before {{ background: var(--accent-alt); }}
    .kpi-card.gold::before {{ background: #f59e0b; }}
    .kpi-label {{
      font-size: 0.85rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--text-main);
    }}
    .kpi-subtext {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 6px;
    }}
    .tabs {{
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
      overflow-x: auto;
    }}
    .tab-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 10px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.95rem;
      transition: all 0.2s ease;
    }}
    .tab-btn:hover {{
      color: var(--text-main);
      background: rgba(255,255,255,0.05);
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: #fff;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }}
    .card-title {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #e2e8f0;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
      gap: 24px;
    }}
    @media (max-width: 768px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}
    .plot-container {{
      background: #0f172a;
      border-radius: 10px;
      padding: 16px;
      border: 1px solid #334155;
      text-align: center;
    }}
    .plot-container img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    th {{
      background: #1e293b;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.05em;
    }}
    tr:hover td {{
      background: rgba(255,255,255,0.02);
    }}
    .tag-cluster-0 {{
      background: rgba(16, 185, 129, 0.18);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.8rem;
    }}
    .tag-cluster-1 {{
      background: rgba(59, 130, 246, 0.18);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.8rem;
    }}
    .terminal-box {{
      background: #090d16;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 18px;
      font-family: Consolas, Monaco, "Courier New", monospace;
      font-size: 0.88rem;
      color: #38bdf8;
      overflow-x: auto;
      white-space: pre-wrap;
      max-height: 480px;
    }}
    .search-bar {{
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .search-input {{
      flex: 1;
      min-width: 240px;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid #475569;
      background: #0f172a;
      color: #fff;
      font-size: 0.95rem;
    }}
    .filter-select {{
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid #475569;
      background: #0f172a;
      color: #fff;
      font-size: 0.95rem;
    }}
    .download-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 14px;
    }}
    .download-card {{
      background: #0f172a;
      border: 1px solid #334155;
      padding: 16px 20px;
      border-radius: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .btn {{
      background: var(--primary);
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      text-decoration: none;
      font-size: 0.85rem;
      transition: background 0.2s;
    }}
    .btn:hover {{
      background: var(--primary-hover);
    }}
    .btn-green {{
      background: var(--accent);
    }}
    .btn-green:hover {{
      background: #059669;
    }}
    .bar-container {{
      margin-bottom: 14px;
    }}
    .bar-label {{
      display: flex;
      justify-content: space-between;
      font-size: 0.88rem;
      margin-bottom: 4px;
    }}
    .bar-track {{
      background: #334155;
      height: 10px;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 6px;
    }}
  </style>
</head>
<body>

  <header>
    <div>
      <h1 style="font-size: 1.55rem; font-weight: 700; margin-bottom: 4px;">Customer Segmentation & Clustering Output</h1>
      <p style="color: var(--text-muted); font-size: 0.9rem;">Machine Learning Pipeline • K-Means (k=2) • 2D PCA Projection • Behavioral Profiling</p>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <div class="badge"><span class="status-dot"></span> Pipeline Run: Complete</div>
      <button class="btn btn-green" onclick="window.print()">Export Report</button>
    </div>
  </header>

  <div class="container">

    <!-- KPI Summary Row -->
    <div class="grid-kpi">
      <div class="kpi-card">
        <div class="kpi-label">Total Customers</div>
        <div class="kpi-value">2,240</div>
        <div class="kpi-subtext">Cleaned & standard scaled dataset</div>
      </div>
      <div class="kpi-card accent">
        <div class="kpi-label">High-Value Segment (Cluster 0)</div>
        <div class="kpi-value">903 (40.3%)</div>
        <div class="kpi-subtext">Avg Spending: <b>$1,233.63</b> • Response: <b>22.6%</b></div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-label">Regular Segment (Cluster 1)</div>
        <div class="kpi-value">1,337 (59.7%)</div>
        <div class="kpi-subtext">Avg Spending: <b>$181.76</b> • Response: <b>9.7%</b></div>
      </div>
      <div class="kpi-card gold">
        <div class="kpi-label">Optimal Clustering</div>
        <div class="kpi-value">k = 2 Clusters</div>
        <div class="kpi-subtext">Silhouette Score: <b>0.3120</b> (Highest)</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('overview')">Executive Summary</button>
      <button class="tab-btn" onclick="switchTab('visualizations')">Model Visualizations (PCA & Silhouette)</button>
      <button class="tab-btn" onclick="switchTab('products-channels')">Product & Channel Profiles</button>
      <button class="tab-btn" onclick="switchTab('campaigns')">Campaign Response & ROI</button>
      <button class="tab-btn" onclick="switchTab('explorer')">Customer Data Explorer</button>
      <button class="tab-btn" onclick="switchTab('terminal')">Execution Logs</button>
    </div>

    <!-- Tab 1: Executive Overview -->
    <div id="overview" class="tab-content active">
      <div class="card">
        <h2 class="card-title">Cluster Profile Comparison</h2>
        <table>
          <thead>
            <tr>
              <th>Cluster</th>
              <th>Segment Name</th>
              <th>Customer Count</th>
              <th>Avg Age</th>
              <th>Avg Income</th>
              <th>Avg Recency</th>
              <th>Avg Spending</th>
              <th>Avg Purchases</th>
              <th>Campaign Response</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><span class="tag-cluster-0">Cluster 0</span></td>
              <td><b>High-Value Customers</b></td>
              <td>903 (40.3%)</td>
              <td>46.9 yrs</td>
              <td><b>$71,339</b></td>
              <td>49.5 days</td>
              <td style="color: #34d399; font-weight: bold;">$1,233.63</td>
              <td>19.7 purchases</td>
              <td><span style="color: #34d399; font-weight: 700;">22.6%</span></td>
            </tr>
            <tr>
              <td><span class="tag-cluster-1">Cluster 1</span></td>
              <td><b>Regular Customers</b></td>
              <td>1,337 (59.7%)</td>
              <td>44.0 yrs</td>
              <td><b>$39,337</b></td>
              <td>48.8 days</td>
              <td style="color: #60a5fa; font-weight: bold;">$181.76</td>
              <td>7.7 purchases</td>
              <td><span style="color: #60a5fa; font-weight: 700;">9.7%</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <h2 class="card-title">Strategic Insights & Recommendations</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px;">
          <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 18px;">
            <h3 style="color: #34d399; margin-bottom: 8px;">Segment 0: High-Value Customers (40.3%)</h3>
            <ul style="padding-left: 20px; font-size: 0.92rem; color: #cbd5e1; line-height: 1.8;">
              <li><b>Spending Power:</b> Outspends regular customers by <b>6.8x</b> ($1,234 vs $182).</li>
              <li><b>Key Categories:</b> Premium Wines ($608 avg) and Meat Products ($359 avg).</li>
              <li><b>Channel Behavior:</b> Heavy in-store (8.6 purchases) and catalog shoppers (5.3 purchases).</li>
              <li><b>Action:</b> Launch VIP concierge perks, early wine reserve releases, and personalized direct mail catalog promotions.</li>
            </ul>
          </div>
          <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 18px;">
            <h3 style="color: #60a5fa; margin-bottom: 8px;">Segment 1: Regular Customers (59.7%)</h3>
            <ul style="padding-left: 20px; font-size: 0.92rem; color: #cbd5e1; line-height: 1.8;">
              <li><b>Engagement Gap:</b> Visit the website <b>6.4 times/month</b> (vs 3.8 for high-value), but have low conversion rates.</li>
              <li><b>Price Sensitivity:</b> Average income is $39,337 with average basket spending of $182.</li>
              <li><b>Action:</b> Use digital cart incentives, entry-level discount bundles, and flash deals to convert frequent web visits into completed checkouts.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 2: Visualizations -->
    <div id="visualizations" class="tab-content">
      <div class="charts-grid">
        <div class="card">
          <h2 class="card-title">K-Means Cluster Selection (Silhouette Score)</h2>
          <div class="plot-container">
            <img src="{b64_silhouette}" alt="K-Means Cluster Selection Silhouette Score">
          </div>
          <p style="font-size: 0.88rem; color: var(--text-muted); margin-top: 12px;">
            Evaluated k=2 through k=8. <b>k=2 achieves the peak silhouette score (0.312)</b>, indicating optimal cluster separation and cohesion.
          </p>
        </div>

        <div class="card">
          <h2 class="card-title">Customer Segmentation – 2D PCA View</h2>
          <div class="plot-container">
            <img src="{b64_pca}" alt="2D PCA Customer Segmentation View">
          </div>
          <p style="font-size: 0.88rem; color: var(--text-muted); margin-top: 12px;">
            PCA reduces multidimensional spending, recency, and demographic features into 2 primary components (explaining <b>52.2%</b> of total variance). Clear separation between High-Value (Cluster 0) and Regular (Cluster 1).
          </p>
        </div>
      </div>
    </div>

    <!-- Tab 3: Products & Channels -->
    <div id="products-channels" class="tab-content">
      <div class="charts-grid">
        <div class="card">
          <h2 class="card-title">Average Spending by Product Category ($)</h2>
          <div style="margin-top: 16px;">
            <div class="bar-container">
              <div class="bar-label"><span>Wines</span> <span>High-Value: <b>$607.63</b> | Regular: <b>$98.82</b></span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width: 86%; background: #10b981;"></div>
                <div class="bar-fill" style="width: 14%; background: #3b82f6;"></div>
              </div>
            </div>
            <div class="bar-container">
              <div class="bar-label"><span>Meat Products</span> <span>High-Value: <b>$358.63</b> | Regular: <b>$37.49</b></span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width: 90%; background: #10b981;"></div>
                <div class="bar-fill" style="width: 10%; background: #3b82f6;"></div>
              </div>
            </div>
            <div class="bar-container">
              <div class="bar-label"><span>Gold Products</span> <span>High-Value: <b>$75.49</b> | Regular: <b>$22.77</b></span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width: 77%; background: #10b981;"></div>
                <div class="bar-fill" style="width: 23%; background: #3b82f6;"></div>
              </div>
            </div>
            <div class="bar-container">
              <div class="bar-label"><span>Fish Products</span> <span>High-Value: <b>$78.75</b> | Regular: <b>$9.68</b></span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width: 89%; background: #10b981;"></div>
                <div class="bar-fill" style="width: 11%; background: #3b82f6;"></div>
              </div>
            </div>
            <div class="bar-container">
              <div class="bar-label"><span>Sweet Products</span> <span>High-Value: <b>$57.54</b> | Regular: <b>$6.48</b></span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width: 90%; background: #10b981;"></div>
                <div class="bar-fill" style="width: 10%; background: #3b82f6;"></div>
              </div>
            </div>
            <div class="bar-container">
              <div class="bar-label"><span>Fruits</span> <span>High-Value: <b>$55.60</b> | Regular: <b>$6.51</b></span></div>
              <div class="bar-track">
                <div class="bar-fill" style="width: 89%; background: #10b981;"></div>
                <div class="bar-fill" style="width: 11%; background: #3b82f6;"></div>
              </div>
            </div>
          </div>
          <div style="display: flex; gap: 18px; margin-top: 18px; font-size: 0.85rem;">
            <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 12px; height: 12px; background: #10b981; border-radius: 2px;"></div> High-Value Customers</div>
            <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 12px; height: 12px; background: #3b82f6; border-radius: 2px;"></div> Regular Customers</div>
          </div>
        </div>

        <div class="card">
          <h2 class="card-title">Channel Preferences & Frequency</h2>
          <table>
            <thead>
              <tr>
                <th>Channel</th>
                <th>High-Value Segment</th>
                <th>Regular Segment</th>
                <th>Dominant Segment</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Store Purchases</td>
                <td><b>8.60</b> / customer</td>
                <td><b>3.89</b> / customer</td>
                <td><span class="tag-cluster-0">High-Value (+121%)</span></td>
              </tr>
              <tr>
                <td>Web Purchases</td>
                <td><b>5.79</b> / customer</td>
                <td><b>2.93</b> / customer</td>
                <td><span class="tag-cluster-0">High-Value (+98%)</span></td>
              </tr>
              <tr>
                <td>Catalog Purchases</td>
                <td><b>5.31</b> / customer</td>
                <td><b>0.87</b> / customer</td>
                <td><span class="tag-cluster-0">High-Value (+510%)</span></td>
              </tr>
              <tr>
                <td>Monthly Web Visits</td>
                <td><b>3.75</b> visits/mo</td>
                <td><b>6.38</b> visits/mo</td>
                <td><span class="tag-cluster-1">Regular (+70% browsing)</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Tab 4: Marketing Campaigns -->
    <div id="campaigns" class="tab-content">
      <div class="card">
        <h2 class="card-title">Marketing Campaign Response & Conversion Rate</h2>
        <table>
          <thead>
            <tr>
              <th>Campaign</th>
              <th>High-Value Acceptance</th>
              <th>Regular Acceptance</th>
              <th>Conversion Ratio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Campaign 1</td>
              <td><b>14.5%</b></td>
              <td>1.1%</td>
              <td><span class="tag-cluster-0">13.2x higher</span></td>
            </tr>
            <tr>
              <td>Campaign 2</td>
              <td><b>3.2%</b></td>
              <td>0.3%</td>
              <td><span class="tag-cluster-0">10.7x higher</span></td>
            </tr>
            <tr>
              <td>Campaign 3</td>
              <td><b>11.2%</b></td>
              <td>4.8%</td>
              <td><span class="tag-cluster-0">2.3x higher</span></td>
            </tr>
            <tr>
              <td>Campaign 4</td>
              <td><b>15.1%</b></td>
              <td>2.3%</td>
              <td><span class="tag-cluster-0">6.6x higher</span></td>
            </tr>
            <tr>
              <td>Campaign 5</td>
              <td><b>16.7%</b></td>
              <td>0.7%</td>
              <td><span class="tag-cluster-0">23.8x higher</span></td>
            </tr>
            <tr style="background: rgba(255,255,255,0.03); font-weight: bold;">
              <td>Final Response Rate</td>
              <td style="color: #34d399;">22.6%</td>
              <td style="color: #60a5fa;">9.7%</td>
              <td><span class="tag-cluster-0">2.3x higher overall</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Tab 5: Customer Data Explorer -->
    <div id="explorer" class="tab-content">
      <div class="card">
        <h2 class="card-title">Interactive Customer Records</h2>
        <div class="search-bar">
          <input type="text" id="searchInput" class="search-input" placeholder="Search customer by ID, Education, or Marital Status..." onkeyup="filterTable()">
          <select id="segmentFilter" class="filter-select" onchange="filterTable()">
            <option value="ALL">All Segments</option>
            <option value="High-Value Customers">High-Value Customers</option>
            <option value="Regular Customers">Regular Customers</option>
          </select>
        </div>
        <div style="overflow-x: auto;">
          <table id="customerTable">
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Age</th>
                <th>Education</th>
                <th>Marital Status</th>
                <th>Income ($)</th>
                <th>Total Spending ($)</th>
                <th>Purchases</th>
                <th>Web Visits</th>
                <th>Segment</th>
                <th>Response</th>
              </tr>
            </thead>
            <tbody id="tableBody">
            </tbody>
          </table>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 14px;">Showing loaded active records. All 2,240 records are exported in `customer_segments_output.csv`.</p>
      </div>
    </div>

    <!-- Tab 6: Terminal Logs -->
    <div id="terminal" class="tab-content">
      <div class="card">
        <h2 class="card-title">Python Code Execution Output (Console Logs)</h2>
        <div class="terminal-box">
$ python customer_segmentation.py

Dataset shape: (2240, 29)
     ID  Year_Birth   Education Marital_Status   Income  ...  Response
0  5524        1957  Graduation         Single  58138.0  ...         1
1  2174        1954  Graduation         Single  46344.0  ...         0
2  4141        1965  Graduation       Together  71613.0  ...         0
3  6182        1984  Graduation       Together  26646.0  ...         0
4  5324        1981         PhD       Together  58293.0  ...         0

[5 rows x 29 columns]

Missing value check:
Income    24
dtype: int64
-> Missing Income imputed with median (51,381.5)

Feature engineering completed:
 - Age = 2014 - Year_Birth
 - Total_Spending = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
 - Total_Purchases = NumWebPurchases + NumCatalogPurchases + NumStorePurchases
 - Total_Household_Children = Kidhome + Teenhome
 - Customer_Tenure_Days = (MaxDate - Dt_Customer).days
 - Preprocessing: Outlier clipping [1%, 99%], StandardScaler, OneHotEncoder

Silhouette scores calculated:
 - k=2: 0.3120  [OPTIMAL]
 - k=3: 0.2203
 - k=4: 0.1672
 - k=5: 0.1016
 - k=6: 0.0987
 - k=7: 0.0988
 - k=8: 0.0989

Selected k: 2 (Silhouette Score: 0.312)

Cluster Profile:
   Cluster               Segment  Customers  AvgAge  AvgIncome  AvgSpending  AvgPurchases  ResponseRate
0        0  High-Value Customers        903   46.95   71339.04      1233.63         19.71          0.23
1        1     Regular Customers       1337   44.01   39337.26       181.76          7.70          0.10

2D PCA Explained Variance:
 - PC1: 41.48%
 - PC2: 10.76%
 - Total Explained: 52.24%

Saved outputs successfully:
 ✓ customer_segments_output.csv
 ✓ segment_profile.csv
 ✓ channel_preferences_by_segment.csv
 ✓ campaign_response_by_segment.csv
 ✓ spend_by_segment.csv
 ✓ dashboard_data.json
 ✓ kmeans_cluster_selection.png
 ✓ customer_segmentation_pca.png
        </div>
      </div>

      <div class="card">
        <h2 class="card-title">Generated Project Files & Datasets</h2>
        <div class="download-grid">
          <div class="download-card">
            <div>
              <div style="font-weight: 600;">customer_segments_output.csv</div>
              <div style="font-size: 0.8rem; color: var(--text-muted);">2,240 rows with cluster & segment labels</div>
            </div>
            <a href="customer_segments_output.csv" download class="btn">Download</a>
          </div>
          <div class="download-card">
            <div>
              <div style="font-weight: 600;">segment_profile.csv</div>
              <div style="font-size: 0.8rem; color: var(--text-muted);">Segment averages & KPIs</div>
            </div>
            <a href="segment_profile.csv" download class="btn">Download</a>
          </div>
          <div class="download-card">
            <div>
              <div style="font-weight: 600;">channel_preferences_by_segment.csv</div>
              <div style="font-size: 0.8rem; color: var(--text-muted);">Web, Store & Catalog purchase habits</div>
            </div>
            <a href="channel_preferences_by_segment.csv" download class="btn">Download</a>
          </div>
          <div class="download-card">
            <div>
              <div style="font-weight: 600;">campaign_response_by_segment.csv</div>
              <div style="font-size: 0.8rem; color: var(--text-muted);">Campaign response & conversion rates</div>
            </div>
            <a href="campaign_response_by_segment.csv" download class="btn">Download</a>
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    const tableData = {json.dumps(table_records)};

    function renderTable(data) {{
      const tbody = document.getElementById("tableBody");
      tbody.innerHTML = "";
      data.forEach(row => {{
        const tr = document.createElement("tr");
        const tagClass = row.Segment === "High-Value Customers" ? "tag-cluster-0" : "tag-cluster-1";
        tr.innerHTML = `
          <td><b>${{row.ID}}</b></td>
          <td>${{row.Age}}</td>
          <td>${{row.Education}}</td>
          <td>${{row.Marital_Status}}</td>
          <td>$${{Number(row.Income).toLocaleString()}}</td>
          <td><b>$${{Number(row.Total_Spending).toLocaleString()}}</b></td>
          <td>${{row.Total_Purchases}}</td>
          <td>${{row.NumWebVisitsMonth}}</td>
          <td><span class="${{tagClass}}">${{row.Segment}}</span></td>
          <td>${{row.Response === 1 ? '<span style="color:#34d399; font-weight:bold;">Yes (1)</span>' : '<span style="color:#94a3b8;">No (0)</span>'}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterTable() {{
      const search = document.getElementById("searchInput").value.toLowerCase();
      const segment = document.getElementById("segmentFilter").value;
      const filtered = tableData.filter(r => {{
        const matchSearch = String(r.ID).includes(search) || 
                            r.Education.toLowerCase().includes(search) || 
                            r.Marital_Status.toLowerCase().includes(search);
        const matchSegment = segment === "ALL" || r.Segment === segment;
        return matchSearch && matchSegment;
      }});
      renderTable(filtered);
    }}

    function switchTab(tabId) {{
      document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
      document.getElementById(tabId).classList.add("active");
      event.target.classList.add("active");
    }}

    // Initial render
    renderTable(tableData);
  </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# Also create dashboard.html copy for convenience
with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Created index.html and dashboard.html successfully.")

