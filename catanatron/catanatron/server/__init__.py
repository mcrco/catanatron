"""A networked, verifiably fair game server on stock Catanatron.

* :mod:`~catanatron.server.app` -- the WebSocket server, lobby and rooms
  (``catanatron-server``), which also serves a browser client;
* :mod:`~catanatron.server.client` -- the async client library;
* :mod:`~catanatron.server.bot` -- registry bots playing from views
  (``catanatron-play-bot``);
* :mod:`~catanatron.server.verify` -- replays a transcript and checks it
  (``catanatron-verify``).

See README.md next to this file for the protocol and the fairness scheme.
"""
