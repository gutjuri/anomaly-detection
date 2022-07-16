# Login_to_vec

This programme is used for extracting logins from /var/log/secure.

Usage: `go run login_to_vec.go /var/log/secure > logins.json`

The programme currently uses the file AS553.json for mapping IP-Addresses to locations.
If a different mapping is desired, the file must be changed.
