import React, { useEffect, useState } from 'react';
import PersonalRoster from './PersonalRoster';
import '../styles/LeagueConnector.css';

export default function LeagueConnector() {
  const [username, setUsername] = useState('');
  const [leagueId, setLeagueId] = useState('');
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [leagueInfo, setLeagueInfo] = useState(null);

  useEffect(() => {
    const savedUsername = localStorage.getItem('sleeper_username');
    const savedLeagueId = localStorage.getItem('sleeper_league_id');
    if (savedUsername && savedLeagueId) {
      setUsername(savedUsername);
      setLeagueId(savedLeagueId);
      setConnected(true);
    }
  }, []);

  const handleConnect = async () => {
    setLoading(true);
    setError(null);

    try {
      const userResponse = await fetch(`https://api.sleeper.app/v1/user/${username}`);
      if (!userResponse.ok) {
        throw new Error('User not found');
      }
      const user = await userResponse.json();
      const userId = user.user_id;

      const leagueResponse = await fetch(`https://api.sleeper.app/v1/league/${leagueId}`);
      if (!leagueResponse.ok) {
        throw new Error('League not found');
      }
      const league = await leagueResponse.json();

      const rostersResponse = await fetch(`https://api.sleeper.app/v1/league/${leagueId}/rosters`);
      if (!rostersResponse.ok) {
        throw new Error('Failed to fetch rosters');
      }
      const rosters = await rostersResponse.json();
      const userRoster = rosters.find((r) => r.owner_id === userId);

      if (!userRoster) {
        throw new Error('Your roster was not found in this league');
      }

      localStorage.setItem('sleeper_username', username);
      localStorage.setItem('sleeper_league_id', leagueId);

      setLeagueInfo({
        leagueName: league.name,
        teamName: userRoster.display_name || userRoster.team_name || 'Your Team',
        userId,
        rosterId: userRoster.roster_id,
      });
      setConnected(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDisconnect = () => {
    setConnected(false);
    setUsername('');
    setLeagueId('');
    setLeagueInfo(null);
    setError(null);
    localStorage.removeItem('sleeper_username');
    localStorage.removeItem('sleeper_league_id');
  };

  if (connected && username && leagueId && leagueInfo) {
    return (
      <div className="league-connector">
        <div className="league-connector__connected-header">
          <div className="league-connector__connection-info">
            <span className="league-connector__league-name">{leagueInfo.leagueName}</span>
            <span className="league-connector__team-name">{leagueInfo.teamName}</span>
            <span className="league-connector__username">@{username}</span>
          </div>
          <button onClick={handleDisconnect} className="league-connector__disconnect-btn">
            Disconnect
          </button>
        </div>

        <PersonalRoster leagueId={leagueId} userId={leagueInfo.userId} />
      </div>
    );
  }

  return (
    <div className="league-connector">
      <div className="league-connector__form">
        <h2>Connect Sleeper League</h2>
        <p>
          See your real Sleeper roster matched against this project&apos;s real fantasy
          projections. Reads directly from Sleeper&apos;s public API in your browser - nothing is
          sent to or stored on any server run by this project.
        </p>

        <div className="league-connector__form-group">
          <label htmlFor="sleeper-username">Sleeper Username</label>
          <input
            id="sleeper-username"
            type="text"
            placeholder="e.g. mahomes_fan"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="league-connector__form-group">
          <label htmlFor="sleeper-league-id">League ID</label>
          <input
            id="sleeper-league-id"
            type="text"
            placeholder="sleeper.app/leagues/[ID]"
            value={leagueId}
            onChange={(e) => setLeagueId(e.target.value)}
            disabled={loading}
          />
        </div>

        <button
          onClick={handleConnect}
          disabled={loading || !username || !leagueId}
          className="league-connector__connect-btn"
        >
          {loading ? 'Connecting...' : 'Connect League'}
        </button>

        {error && <div className="league-connector__error-message">{error}</div>}

        <div className="league-connector__instructions">
          <h3>How to find your League ID</h3>
          <ol>
            <li>
              Open your league at{' '}
              <a href="https://sleeper.app" target="_blank" rel="noopener noreferrer">
                sleeper.app
              </a>
            </li>
            <li>
              Copy the ID from the URL: <code>sleeper.app/leagues/[ID]</code>
            </li>
            <li>Paste it above</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
