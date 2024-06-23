@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos
set ACCOUNT_ID=697426589657

:: Attach the policy to the user
aws iam attach-user-policy --user-name SystemDeveloper-Xellos --policy-arn arn:aws:iam::%ACCOUNT_ID%:policy/GlueFullAccessPolicy --profile %AWS_PROFILE%

endlocal
pause
