# request validation
- **Workflow trigger**
```yaml
      - name: Pre Processing
        id: pre_process_validation
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "Extracting game name"
          jq -r '.issue.body' "$GITHUB_EVENT_PATH" > /tmp/game_name.txt
          echo "DEBUG: validate game_name"
          cat /tmp/game_name.txt
          echo "DEBUG: Load github output"
          echo "game_name=$(cat /tmp/game_name.txt)" >> $GITHUB_OUTPUT
          echo "DEBUG: Test github outputs"
          echo $GITHUB_OUTPUT

      - name: Validate Request
        uses: drewpypro/drewpy-actions/request_validation@v1
        id: request_validation
        with:
          game_name: ${{ steps.pre_process_validation.outputs.game_name }}
```
- **drewpy-actions/request_validation.py**
    - Accepts --input-file to receive $game_name variable
    - takes issue request input and performs linting, validation, sanitization and rejection. 
        - if character count is over 69, then reject the request. **too long**
        - if non-standard unicode character is present, request the request right away and return back to user. **bad characters**
            - standard unicode acceppted for [A-Z],[a-z],[0-9]
            - accepted characters: spaces, "!", "?", ",", ":", "'", "-", "&", ".", "™" (Unicode: U+2122), "_", open/closed parenthesis.
                - "-" and "." do not get transformed.
                - "&" gets transformed to "-and-"
                - all other accepted characters get translated to "-"
        - if game_name="ARC Raiders", then it gets normalized to "arc-raiders", Insurgency: Sandstorm, is normalized into "insurgency-sandstorm"
        - game_name fuzzy matches "ARK*.*Survival" case-insensitive, then drop if match. **access denied**
    - if successful provide "Validation Okay" response, returns failure codes; 
        - **script error** - not returned back to issue, generated and output to pipeline excluding secrets/sensitive information. 
            - Output Message: Script has errored, please review logs and contact admin.
            - Output message is displayed in pipeline only. 
        - **too long** - starting with 69 characters or less, but need to assess common/longest game names in my library to make sure that works.
            - Output Message: Game name longer than 69 characters, please shorten and try again or contact the administrator.
            - Output message is emitted back to issue/user. 
        - **bad characters** - using non-standard unicode characters or unsupported special characters.
            - Output Message: Bad characters in your request, please try again by submitting again using a normal human keyboard.
            - output message is emitted back to issue/user. 
        - **access denied** - create a list of games and refuse assessment, or if a game pattern matches something nsfw for example.
            - Output Message: I refuse to assess this game, it sucks. 
        - If successful, generates json and md file to $GITHUB_OUTPUT.
            - $repo/json/$slug-assessment.json
            - $repo/docs/$slug-assessment.md

