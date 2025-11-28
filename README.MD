# Drewpy Actions
![drewpy-actions](./img/tribird.gif)

```
game_assessments/
└── .github/
    ├── workflows/
    │   ├── generate_ai_assessment.yaml    // calls apis and creates files and handles PRs
    │   └── docs_updater.yaml             // After PR is approved generates docs page in drewpypro.github.io
    └── ISSUE_TEMPLATE/
        └── game_assessor_reques.yaml   // Issue template that triggers pipeline and PR. 
├── output/$game_name.json             // JSON entry per game
└── docs/$game_name.md                // Markdown formatted report pushed to drewpypro.github.io
```

# Game Assessment workflow (drewpypro/game_assessments)
1. validate workflow
    * user submits issue form with a game name (.github/ISSUE_TEMPLATE/generate-game-assessment.yaml)
    * git action extracts $game_name into variable used in next steps
    * linting occurs (standard unicode characters, no length greater than 69, transform name to standard (i.e., "ARC Raiders", turns into arc-raiders)) using drewpy-actions/request_validation.py
        * If script emits "validation okay", proceed. 
        * If script emits error code, respond back to the user with the preferred error language. 
        * rejection is returned back to the users issue with comments for rejection reason. 
    * creates $game_name-assessment.json empty file; 
    * If $author !=drewpypro, create PR and stop workflow. ⚠️
    * if approved PR that retriggers this pipeline, ignore this step.
    * if $author = drewpypro, continue to assessment workflow.
2. assessment workflow
    * if $AUTHOR is drewpypro, or a PR approval triggered this workflow, runs composite action from drewpy-ai-actions/call_assessment_api.py to trigger researcher response and generate /assessments/$game_name-assessment.json
    * if successful, then creates /assessments/$game_name-assessment.json and /docs/$game_name-assessment.md
    * Creates PR on issue-branch with both files for approval. ⚠️
3. Approved assessment workflow
    * PR is approved and merged to main
    * gh commit and push /docs/$game_name-assessment.md to drewpypro.github.io/docs/assessments/$game_name-assessment.md on staging branch.

# Docs Building workflow (drewpypro/drewpypro.github.io)
1. generate docs
    * new commit to staging branch retriggers docusaurus build and yarn deploy to gh-pages branch. 

# Git Actions 
- **drewpy-actions/request_validation.py**
    - Accepts --input-file to receive $game_name variable
    - takes issue request input and performs linting, validation, sanitization and rejection. 
    - if successful provide "Validation Okay" response, returns failure codes; 
        - **script error** - not returned back to issue, generated and output to pipeline excluding secrets/sensitive information. 
        - **too long** - starting with 69 characters or less, but need to assess common/longest game names in my library to make sure that works. 
        - **bad characters** - using non-standard unicode characters or unsupported special characters. 
        - **access denied** - create a list of games and refuse assessment, or if a game pattern matches something nsfw for example. 
- **drewpy-ai-actions/call_assessment_api.py**
    - where assessment instructions, context and persona are stored. 
    - generates game assessment json db. 
- **drewpy-actions/report_generator.py**
    - takes input $game_name-assessment.json file and generates markdown report based on template. 
- **drewpy-actions/doc_generator.py**
    - receives input pushed to staging branch and generates required npm and yarn commands required to build the site and deploy the docs on drewpypro.github.io through the gh-pages branch. 

# To Do
- rule input sanitization (working on in scripts/request_validation.py)
- cron scheduler that regenerates assessments after a certain period of time. 
- create more detailed questions, gather more requirements
    - add image url retrieval for docs.
- generate new ai-images for repos. 
- start testing workflow and scripts



```
game_assessments/
└── .github/
    ├── workflows/
    │   ├── generate_ai_assessment.yaml    // calls apis and creates files and handles PRs
    │   └── docs_updater.yaml             // After PR is approved generates docs page in drewpypro.github.io
    └── ISSUE_TEMPLATE/
        └── game_assessor_reques.yaml   // Issue template that triggers pipeline and PR. 
├── output/$game_name.json             // JSON entry per game
└── docs/$game_name.md                // Markdown formatted report pushed to drewpypro.github.io
```

```
drewpy-ai-actions/
└── call_assessment_api/
    ├── call_assessment_api.py           // calls apis and generates docs files
    ├── README.MD
    ├── requirements.txt
    └── action.yml                       
```

```
drewpy-actions/
├── request_validation/
    ├── request_validation.py           // linting occurs (standard unicode characters, no length greater than 69, transform name to standard (i.e., "ARC Raiders", turns into arc-raiders))
    ├── README.MD
    ├── requirements.txt
    └── action.yml  
```