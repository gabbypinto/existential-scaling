## SSH Config

Edit `~/.ssh/config` with you editor of choice.  
I just use vim so:  
`vim ~/.ssh/config`

If it doesn't exist, you need to create it:

`touch ~/.ssh/config`  
`chmod 600 ~/.ssh/config`

Create an entry for the MLAT 07 Cluster
Edit out the username (User) to your username 
(you can also rename Host to anything you want, I just like this for my config)

```
Host mlat_cluster_07
    Hostname 192.168.200.37
    User spencer
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

You may also need to generate an SSH key if you haven't done that before:

`ssh-keygen -t ed25519 -C "your_email@example.com"`

Then copy the ssh key onto the cluster

`ssh-copy-id -i ~/.ssh/id_ed25519.pub mlat_cluster_07`

Now you can try for a quick test via:  
`ssh mlat_cluster_07`  
to see if you can connect *(note: may take a bit longer than other ssh stuff)*