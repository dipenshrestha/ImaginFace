
#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r ./src/requirements.txt

python ./src/manage.py collectstatic --no-input
python ./src/manage.py migrate
if [[ $CREATE_SUPERUSER ]];
then
  python ./src/manage.py createsuperuser --no-input --email "$DJANGO_SUPERUSER_EMAIL"
fi