@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos
set ACCOUNT_ID=697426589657

:: Create the policy
aws iam create-policy --policy-name GlueFullAccessForUser --policy-document file://glue-user-policy.json --profile %AWS_PROFILE%

:: Attach the policy to the user
aws iam attach-user-policy --user-name SystemDeveloper-Xellos --policy-arn arn:aws:iam::%ACCOUNT_ID%:policy/GlueFullAccessForUser --profile %AWS_PROFILE%

endlocal
pause
