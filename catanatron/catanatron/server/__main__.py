import sys

from catanatron.server.app import main

# Guarded: bot worker processes import the main module too.
if __name__ == "__main__":
    sys.exit(main())
