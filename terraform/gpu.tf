resource "aws_spot_instance_request" "spot_gpu_worker" {
  ami           = "ami-00e58ffd73e16170c"
  spot_price    = "${var.GPU_SPOT_PRICE}"
  instance_type = "${var.GPU_INSTANCE_TYPE}"
  key_name = "${var.AWS_KEY_NAME}"
  user_data = <<EOF
  #! /bin/bash
  cd /home/ec2-user
  git clone https://github.com/silversmith123/IntuitionShogi.git
  sudo -u ec2-user /home/ec2-user/anaconda3/envs/tensorflow2_latest_p37/bin/pip install -r IntuitionShogi/requirements.txt
  sudo curl -L https://github.com/kahing/goofys/releases/latest/download/goofys -o /usr/local/bin/goofys
  sudo chmod a+x /usr/local/bin/goofys
  export AWS_ACCESS_KEY_ID="${var.AWS_ACCESS_KEY_ID}"
  export AWS_SECRET_ACCESS_KEY="${var.AWS_SECRET_ACCESS_KEY}"
  export AWS_DEFAULT_REGION="${var.AWS_DEFAULT_REGION}"
  mkdir /home/ec2-user/IntuitionShogi/hcpe/mount
  /usr/local/bin/goofys -o allow_other --uid 1000 --gid 1000 "${var.AWS_S3_BUCKET}" /home/ec2-user/IntuitionShogi/hcpe/mount
  chown -R ec2-user:ec2-user IntuitionShogi
  EOF

  tags = {
    Name = "spoGPUWorker"
  }
}
