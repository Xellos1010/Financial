@echo off
setlocal

:: Set AWS profile
set AWS_PROFILE=SystemDeveloper-Xellos

:: Set the role name
set ROLE_NAME=GlueServiceRole

:: Create IAM role with the trust policy
aws iam create-role --role-name %ROLE_NAME% --assume-role-policy-document file://trust-policy.json --profile %AWS_PROFILE%

:: Attach policies to the role
aws iam attach-role-policy --role-name %ROLE_NAME% --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole --profile %AWS_PROFILE%
aws iam attach-role-policy --role-name %ROLE_NAME% --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --profile %AWS_PROFILE%

endlocal
pause
