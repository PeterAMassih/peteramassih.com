---
title: "CryptKVS — Encrypted Key-Value Store in C"
description: "Client-side-encrypted key-value store in C with both local-file and HTTP-server modes. HMAC-SHA256 password hardening, per-entry random salt. The server cannot decrypt."
pubDate: 2022-05-30
tags: ["C", "cryptography", "systems", "OpenSSL"]
---

## What this is

Encrypted key-value store written from scratch in C. Two backends: a local file and a libmongoose HTTP server. The design is based on Signal's "secure value recovery": the server stores ciphertexts and per-entry salts but never sees plaintext.

## How it works

The crypto:

- HMAC-SHA256 password hardening gives an `auth_key` (sent to the server to authenticate) and `c1` (master-key-encryption key, never leaves the client).
- Each entry stores a random 32-byte `c2` alongside its ciphertext.
- Per-entry AES key = `c1 ⊕ c2`. The server can hand back any entry but can't decrypt it.

Nine C modules: CLI dispatcher, client commands, local backend, RPC client, libmongoose HTTP server, crypto, IO, utilities, errors. One binary `cryptkvs` works against either a local file or a remote server. Built with AddressSanitizer and strict warnings. Per-week unit tests plus end-to-end tests.

CS-212 Programmation Système at EPFL.
