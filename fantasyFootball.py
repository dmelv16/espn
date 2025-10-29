import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(page_title="ManagerIQ Dashboard", layout="wide", page_icon="🏈")

# Database connection
@st.cache_resource
def get_connection():
    """Create database connection - UPDATE WITH YOUR CONNECTION STRING"""
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"  # or {SQL Server} for older driver
        "SERVER=DESKTOP-J9IV3OH;"  # e.g., localhost, 192.168.1.100, or SERVER\INSTANCE
        "DATABASE=espn;"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)

@st.cache_data
def load_data():
    """Load all tables from database"""
    conn = get_connection()
    
    # Load tables - use proper column names from your schema
    teams = pd.read_sql("SELECT * FROM [espn].[dbo].[Teams]", conn)
    standings = pd.read_sql("SELECT * FROM [espn].[dbo].[Standings]", conn)
    matchups = pd.read_sql("SELECT * FROM [espn].[dbo].[Matchups]", conn)
    lineups = pd.read_sql("SELECT * FROM [espn].[dbo].[Lineups]", conn)
    
    # Debug: Print column names to verify
    print("Teams columns:", teams.columns.tolist())
    print("Matchups columns:", matchups.columns.tolist())
    print("Lineups columns:", lineups.columns.tolist())
    
    return teams, standings, matchups, lineups

# Load data
try:
    teams, standings, matchups, lineups = load_data()
except Exception as e:
    st.error(f"⚠️ Database connection error: {e}")
    st.info("💡 Update the connection string in `get_connection()` with your SQL Server details")
    st.stop()

# Title
st.title("🏈 ManagerIQ: Fantasy League Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Manager Report Card",
    "Head-to-Head Analysis",
    "Luck Analyzer",
    "Strength of Schedule",
    "Projection Accuracy",
    "League Overview"
])

# ===========================
# PAGE 1: MANAGER REPORT CARD
# ===========================
if page == "Manager Report Card":
    st.header("📊 Manager Report Card")
    st.markdown("Analyze how well each manager sets their lineup each week")
    
    # Lineups table already has season, week, and team_id - no merge needed!
    lineup_analysis = lineups.copy()
    
    # Points left on bench
    bench_points = lineup_analysis[lineup_analysis['is_starter'] == 0].groupby(
        ['team_id', 'season', 'week']
    )['points_scored'].sum().reset_index()
    bench_points.columns = ['team_id', 'season', 'week', 'bench_points']
    
    # Starter points
    starter_points = lineup_analysis[lineup_analysis['is_starter'] == 1].groupby(
        ['team_id', 'season', 'week']
    )['points_scored'].sum().reset_index()
    starter_points.columns = ['team_id', 'season', 'week', 'starter_points']
    
    # Merge with teams (keep only unique team_id + season combinations)
    teams_unique = teams[['team_id', 'season', 'owner', 'team_name']].drop_duplicates()
    
    manager_stats = bench_points.merge(starter_points, on=['team_id', 'season', 'week'], how='outer')
    manager_stats = manager_stats.merge(teams_unique, on=['team_id', 'season'])
    
    # Fill any NaN values with 0
    manager_stats['bench_points'] = manager_stats['bench_points'].fillna(0)
    manager_stats['starter_points'] = manager_stats['starter_points'].fillna(0)
    
    # Calculate average bench points per manager (across ALL seasons)
    manager_summary = manager_stats.groupby('owner').agg({
        'bench_points': 'mean',
        'starter_points': 'mean',
        'week': 'count'
    }).reset_index()
    manager_summary.columns = ['Owner', 'Avg Bench Points', 'Avg Starter Points', 'Weeks']
    
    # Calculate total bench points
    total_bench = manager_stats.groupby('owner')['bench_points'].sum().reset_index()
    total_bench.columns = ['Owner', 'Total Bench Points']
    manager_summary = manager_summary.merge(total_bench, on='Owner')
    
    manager_summary = manager_summary.sort_values('Avg Bench Points', ascending=False)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Manager (Lowest Bench Pts)", 
                manager_summary.iloc[-1]['Owner'], 
                f"{manager_summary.iloc[-1]['Avg Bench Points']:.1f} pts/wk")
    col2.metric("Worst Manager (Highest Bench Pts)", 
                manager_summary.iloc[0]['Owner'], 
                f"{manager_summary.iloc[0]['Avg Bench Points']:.1f} pts/wk")
    col3.metric("League Average", 
                f"{manager_summary['Avg Bench Points'].mean():.1f} pts/wk")
    
    # Bar chart
    fig = px.bar(manager_summary, x='Owner', y='Avg Bench Points',
                 title='Average Points Left on Bench by Manager (All Seasons)',
                 color='Avg Bench Points',
                 color_continuous_scale='Reds')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Manager Statistics")
    st.dataframe(manager_summary.style.format({
        'Avg Bench Points': '{:.2f}',
        'Avg Starter Points': '{:.2f}',
        'Total Bench Points': '{:.2f}'
    }), use_container_width=True)
    
    # Weekly trends
    st.subheader("Weekly Bench Points Trend")
    selected_owner = st.selectbox("Select Manager", sorted(manager_stats['owner'].unique()))
    
    owner_weekly = manager_stats[manager_stats['owner'] == selected_owner].copy()
    owner_weekly = owner_weekly.sort_values(['season', 'week'])
    
    fig2 = px.line(owner_weekly, x='week', y='bench_points', color='season',
                   title=f'Bench Points by Week - {selected_owner}',
                   markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# ===========================
# PAGE 2: HEAD-TO-HEAD
# ===========================
elif page == "Head-to-Head Analysis":
    st.header("⚔️ Head-to-Head Analysis")
    
    col1, col2 = st.columns(2)
    owners = teams['owner'].unique()
    
    with col1:
        owner1 = st.selectbox("Select Manager 1", owners, key='owner1')
    with col2:
        owner2 = st.selectbox("Select Manager 2", owners, key='owner2')
    
    if owner1 == owner2:
        st.warning("Please select two different managers")
    else:
        # Get team IDs
        team1_ids = teams[teams['owner'] == owner1]['team_id'].values
        team2_ids = teams[teams['owner'] == owner2]['team_id'].values
        
        # Get matchups between these teams
        h2h = matchups[
            ((matchups['team_id'].isin(team1_ids)) & (matchups['opponent_id'].isin(team2_ids))) |
            ((matchups['team_id'].isin(team2_ids)) & (matchups['opponent_id'].isin(team1_ids)))
        ].copy()
        
        # Determine winner
        h2h['winner'] = h2h.apply(lambda x: owner1 if x['team_id'] in team1_ids and x['team_score'] > x['opponent_score']
                                   else owner2 if x['team_id'] in team2_ids and x['team_score'] > x['opponent_score']
                                   else owner2 if x['opponent_id'] in team2_ids and x['team_score'] > x['opponent_score']
                                   else owner1 if x['opponent_id'] in team1_ids and x['team_score'] > x['opponent_score']
                                   else 'Tie', axis=1)
        
        # Calculate stats
        owner1_wins = len(h2h[h2h['winner'] == owner1])
        owner2_wins = len(h2h[h2h['winner'] == owner2])
        ties = len(h2h[h2h['winner'] == 'Tie'])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{owner1} Wins", owner1_wins)
        col2.metric(f"{owner2} Wins", owner2_wins)
        col3.metric("Ties", ties)
        col4.metric("Total Matchups", len(h2h))
        
        # Win pie chart
        if len(h2h) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=[owner1, owner2],
                values=[owner1_wins, owner2_wins],
                hole=.3
            )])
            fig.update_layout(title="Head-to-Head Record")
            st.plotly_chart(fig, use_container_width=True)
            
            # Game history
            st.subheader("Game History")
            h2h_display = h2h[['season', 'week', 'team_score', 'opponent_score', 'winner']].copy()
            h2h_display = h2h_display.sort_values(['season', 'week'], ascending=False)
            st.dataframe(h2h_display, use_container_width=True)

# ===========================
# PAGE 3: LUCK ANALYZER
# ===========================
elif page == "Luck Analyzer":
    st.header("🍀 Luck Analyzer")
    st.markdown("Identify lucky and unlucky teams based on points vs. record")
    
    # Calculate expected wins based on points
    luck_data = standings.merge(teams[['team_id', 'owner', 'season']], on=['team_id', 'season'])
    
    # Win percentage
    luck_data['win_pct'] = luck_data['wins'] / (luck_data['wins'] + luck_data['losses'])
    
    # Calculate expected win percentage based on points (simple approach)
    # More points for = higher expected win %
    season_select = st.selectbox("Select Season", sorted(luck_data['season'].unique(), reverse=True))
    season_data = luck_data[luck_data['season'] == season_select].copy()
    
    # Normalize points for to 0-1 scale
    max_pf = season_data['points_for'].max()
    min_pf = season_data['points_for'].min()
    season_data['expected_win_pct'] = (season_data['points_for'] - min_pf) / (max_pf - min_pf)
    
    # Luck index (positive = lucky, negative = unlucky)
    season_data['luck_index'] = (season_data['win_pct'] - season_data['expected_win_pct']) * 100
    season_data = season_data.sort_values('luck_index', ascending=False)
    
    # Display
    fig = px.bar(season_data, x='owner', y='luck_index',
                 title=f'Luck Index - {season_select} Season',
                 color='luck_index',
                 color_continuous_scale='RdYlGn',
                 labels={'luck_index': 'Luck Index (%)'},
                 hover_data=['wins', 'losses', 'points_for'])
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    fig2 = px.scatter(season_data, x='points_for', y='wins',
                      text='owner', title='Points For vs. Wins',
                      labels={'points_for': 'Total Points For', 'wins': 'Wins'},
                      color='luck_index',
                      color_continuous_scale='RdYlGn')
    fig2.update_traces(textposition='top center', marker=dict(size=12))
    st.plotly_chart(fig2, use_container_width=True)
    
    st.dataframe(season_data[['owner', 'wins', 'losses', 'points_for', 'points_against', 'luck_index']].style.format({
        'points_for': '{:.2f}',
        'points_against': '{:.2f}',
        'luck_index': '{:.2f}'
    }), use_container_width=True)

# ===========================
# PAGE 4: STRENGTH OF SCHEDULE
# ===========================
elif page == "Strength of Schedule":
    st.header("💪 Strength of Schedule Analysis")
    st.markdown("Analyze how difficult each manager's schedule was")
    
    season_select = st.selectbox("Select Season", sorted(matchups['season'].unique(), reverse=True))
    
    # Get all matchups for the season
    season_matchups = matchups[matchups['season'] == season_select].copy()
    
    # Get teams that played in this season only
    season_teams = teams[teams['season'] == season_select][['team_id', 'owner', 'team_name']].drop_duplicates()
    season_team_ids = season_teams['team_id'].unique()
    
    # Filter matchups to only include teams from this season
    season_matchups = season_matchups[season_matchups['team_id'].isin(season_team_ids)]
    
    # Calculate average opponent score for each team
    sos_data = season_matchups.groupby('team_id').agg({
        'opponent_score': ['mean', 'sum', 'count']
    }).reset_index()
    sos_data.columns = ['team_id', 'avg_opp_score', 'total_opp_score', 'games']
    
    # Get team's own average score
    team_scores = season_matchups.groupby('team_id')['team_score'].mean().reset_index()
    team_scores.columns = ['team_id', 'avg_team_score']
    
    # Merge with standings and team info
    sos_data = sos_data.merge(team_scores, on='team_id')
    sos_data = sos_data.merge(
        standings[standings['season'] == season_select][['team_id', 'wins', 'losses', 'points_for']], 
        on='team_id'
    )
    sos_data = sos_data.merge(season_teams, on='team_id')
    
    # Calculate SOS rank (higher avg opponent score = harder schedule)
    sos_data['sos_rank'] = sos_data['avg_opp_score'].rank(ascending=False)
    sos_data = sos_data.sort_values('avg_opp_score', ascending=False)
    
    # Calculate "expected wins" based on if they played league average schedule
    league_avg_opp = sos_data['avg_opp_score'].mean()
    sos_data['schedule_difficulty'] = ((sos_data['avg_opp_score'] - league_avg_opp) / league_avg_opp * 100)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    hardest = sos_data.iloc[0]
    easiest = sos_data.iloc[-1]
    
    col1.metric("Hardest Schedule", hardest['owner'], 
                f"{hardest['avg_opp_score']:.2f} pts/game")
    col2.metric("Easiest Schedule", easiest['owner'], 
                f"{easiest['avg_opp_score']:.2f} pts/game")
    col3.metric("League Avg Opponent Score", f"{league_avg_opp:.2f} pts")
    
    st.markdown("---")
    
    # Visualization 1: Schedule Difficulty Bar Chart
    st.subheader("Schedule Difficulty Rating")
    fig1 = px.bar(sos_data, x='owner', y='schedule_difficulty',
                  title='Schedule Difficulty vs. League Average',
                  color='schedule_difficulty',
                  color_continuous_scale='RdYlGn_r',
                  labels={'schedule_difficulty': 'Difficulty (%)'},
                  hover_data=['avg_opp_score', 'wins', 'losses'])
    fig1.add_hline(y=0, line_dash="dash", line_color="gray", 
                   annotation_text="League Average")
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Visualization 2: Own Score vs Opponent Score
    st.subheader("Team Performance vs Schedule Strength")
    fig2 = px.scatter(sos_data, x='avg_opp_score', y='avg_team_score',
                      text='owner', size='wins', color='wins',
                      title='Your Score vs. Opponent Strength',
                      labels={
                          'avg_opp_score': 'Avg Opponent Score (Schedule Difficulty)',
                          'avg_team_score': 'Your Avg Score',
                          'wins': 'Wins'
                      },
                      color_continuous_scale='Viridis')
    
    # Add diagonal line (where team score = opponent score)
    min_val = min(sos_data['avg_opp_score'].min(), sos_data['avg_team_score'].min())
    max_val = max(sos_data['avg_opp_score'].max(), sos_data['avg_team_score'].max())
    fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                              mode='lines', name='Break Even',
                              line=dict(dash='dash', color='gray')))
    fig2.update_traces(textposition='top center', selector=dict(mode='markers+text'))
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info("📊 **Above the line** = You scored more than your opponents on average\n\n"
            "📊 **Right side** = You faced tougher opponents (harder schedule)")
    
    # Weekly opponent strength visualization
    st.subheader("Weekly Schedule Analysis")
    selected_owner = st.selectbox("Select Manager", sorted(sos_data['owner'].unique()))
    
    selected_team = season_teams[season_teams['owner'] == selected_owner]
    if len(selected_team) > 0:
        team_id = selected_team['team_id'].values[0]
        team_weekly = season_matchups[season_matchups['team_id'] == team_id].copy()
        team_weekly = team_weekly.sort_values('week')
        
        # Add league average for comparison
        weekly_avg = season_matchups.groupby('week')['opponent_score'].mean().reset_index()
        weekly_avg.columns = ['week', 'league_avg_opp']
        team_weekly = team_weekly.merge(weekly_avg, on='week')
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=team_weekly['week'], 
            y=team_weekly['opponent_score'],
            mode='lines+markers',
            name='Your Opponent Score',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        fig3.add_trace(go.Scatter(
            x=team_weekly['week'], 
            y=team_weekly['league_avg_opp'],
            mode='lines',
            name='League Avg Opponent',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig3.add_trace(go.Scatter(
            x=team_weekly['week'], 
            y=team_weekly['team_score'],
            mode='lines+markers',
            name='Your Score',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        fig3.update_layout(
            title=f"Week-by-Week Schedule Strength - {selected_owner}",
            xaxis_title="Week",
            yaxis_title="Points",
            hovermode='x unified'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Detailed table
    st.subheader("Complete Schedule Strength Rankings")
    display_cols = ['sos_rank', 'owner', 'team_name', 'wins', 'losses', 
                   'avg_team_score', 'avg_opp_score', 'schedule_difficulty']
    display_df = sos_data[display_cols].copy()
    display_df.columns = ['SOS Rank', 'Owner', 'Team', 'Wins', 'Losses', 
                          'Avg Score', 'Avg Opp Score', 'Difficulty (%)']
    
    st.dataframe(display_df.style.format({
        'Avg Score': '{:.2f}',
        'Avg Opp Score': '{:.2f}',
        'Difficulty (%)': '{:.2f}',
        'SOS Rank': '{:.0f}'
    }).background_gradient(subset=['Difficulty (%)'], cmap='RdYlGn_r'),
    use_container_width=True)
    
    # Insights
    st.subheader("💡 Key Insights")
    
    # Find teams that performed well despite hard schedule
    sos_data['performance_index'] = sos_data['wins'] / (sos_data['schedule_difficulty'] / 10 + 10)
    overperformer = sos_data.nlargest(1, 'performance_index').iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**🏆 Best Performance vs Schedule**\n\n"
                  f"{overperformer['owner']} went {int(overperformer['wins'])}-{int(overperformer['losses'])} "
                  f"despite facing {abs(overperformer['schedule_difficulty']):.1f}% "
                  f"{'harder' if overperformer['schedule_difficulty'] > 0 else 'easier'} "
                  f"than average schedule")
    
    with col2:
        # Find biggest schedule disparity
        schedule_range = sos_data['avg_opp_score'].max() - sos_data['avg_opp_score'].min()
        st.warning(f"**⚖️ Schedule Fairness**\n\n"
                  f"There was a {schedule_range:.2f} points/game difference "
                  f"between the easiest and hardest schedules this season")

# ===========================
# PAGE 5: PROJECTION ACCURACY
# ===========================
elif page == "Projection Accuracy":
    st.header("🎯 Projection vs. Reality")
    st.markdown("How accurate are ESPN's projections?")
    
    # Calculate projection error
    proj_data = lineups.copy()
    proj_data['error'] = proj_data['points_scored'] - proj_data['projected_points']
    proj_data['abs_error'] = abs(proj_data['error'])
    proj_data['pct_error'] = ((proj_data['points_scored'] - proj_data['projected_points']) / 
                               proj_data['projected_points'].replace(0, 1) * 100)
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", f"{proj_data['abs_error'].mean():.2f} pts")
    col2.metric("Average Projection", f"{proj_data['projected_points'].mean():.2f} pts")
    col3.metric("Average Actual", f"{proj_data['points_scored'].mean():.2f} pts")
    
    # By position
    st.subheader("Projection Accuracy by Position")
    pos_accuracy = proj_data.groupby('position').agg({
        'abs_error': 'mean',
        'projected_points': 'mean',
        'points_scored': 'mean',
        'player_id': 'count'
    }).reset_index()
    pos_accuracy.columns = ['Position', 'Avg Abs Error', 'Avg Projected', 'Avg Actual', 'Count']
    pos_accuracy = pos_accuracy.sort_values('Avg Abs Error', ascending=False)
    
    fig = px.bar(pos_accuracy, x='Position', y='Avg Abs Error',
                 title='Average Projection Error by Position',
                 color='Avg Abs Error',
                 color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot - Projected vs Actual
    st.subheader("Projected vs. Actual Points")
    sample_size = st.slider("Sample size", 100, 5000, 1000)
    sample = proj_data[proj_data['is_starter'] == True].sample(min(sample_size, len(proj_data)))
    
    fig2 = px.scatter(sample, x='projected_points', y='points_scored',
                      color='position', hover_data=['player_name', 'week'],
                      title='Projected vs Actual Points (Starters Only)',
                      opacity=0.6)
    fig2.add_trace(go.Scatter(x=[0, 40], y=[0, 40], mode='lines',
                              name='Perfect Projection', line=dict(dash='dash', color='gray')))
    st.plotly_chart(fig2, use_container_width=True)

# ===========================
# PAGE 6: LEAGUE OVERVIEW
# ===========================
else:
    st.header("🏆 League Overview")
    
    # Season selector
    season = st.selectbox("Select Season", sorted(standings['season'].unique(), reverse=True))
    
    # Get unique teams for this season
    season_teams = teams[teams['season'] == season][['team_id', 'owner', 'team_name']].drop_duplicates()
    
    season_standings = standings[standings['season'] == season].merge(
        season_teams, on='team_id'
    )
    season_standings = season_standings.sort_values('standing')
    
    # Display standings
    st.subheader(f"{season} Season Standings")
    st.dataframe(season_standings[['standing', 'owner', 'team_name', 'wins', 'losses', 
                                   'points_for', 'points_against']].style.format({
        'points_for': '{:.2f}',
        'points_against': '{:.2f}'
    }), use_container_width=True)
    
    # Points distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(season_standings, x='owner', y='points_for',
                      title='Total Points For',
                      color='points_for',
                      color_continuous_scale='Blues')
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(season_standings, x='owner', y='points_against',
                      title='Total Points Against',
                      color='points_against',
                      color_continuous_scale='Reds')
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Weekly scoring trends
    st.subheader("Weekly Scoring Trends")
    weekly_scores = matchups[matchups['season'] == season].merge(
        season_teams, on='team_id'
    )
    
    fig3 = px.line(weekly_scores, x='week', y='team_score', color='owner',
                   title=f'Weekly Scores - {season} Season',
                   markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*ManagerIQ Dashboard - Built with Streamlit & Python*")