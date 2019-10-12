package com.example.homework;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.HideReturnsTransformationMethod;
import android.text.method.PasswordTransformationMethod;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import org.w3c.dom.Text;

public class LoginActivity extends AppCompatActivity {
    private EditText accountEdit;
    private EditText passwordEdit;
    private Button login;
//  private Button register;
    private ToggleButton passwordVisibility;
    private TextView pressRegister;
    @Override
    protected void onActivityResult(int requestCode,int resultCode,Intent data){
        switch(requestCode){
            case 1:
                if(requestCode==RESULT_OK){
                    String returnedData=data.getStringExtra("data_return ");
                    //Log.d
                }
                break;
                default:
        }
    }
    @Override
    public                                                                                                                                                                                                                                                                                     void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        accountEdit=(EditText)findViewById(R.id.account);
        passwordEdit=(EditText)findViewById(R.id.password);
        this.passwordVisibility = (ToggleButton) findViewById(R.id.password_visibility);
        pressRegister=findViewById(R.id.press_to_register);
        passwordEdit.setTransformationMethod(PasswordTransformationMethod.getInstance());
        //事件注册
        this.passwordVisibility.setOnCheckedChangeListener(new ToggleButtonClick());
        login=(Button)findViewById(R.id.login);
        login.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                String account=accountEdit.getText().toString();
                String password=passwordEdit.getText().toString();
                if(account.equals("1")&& password.equals("1")){
                    Intent intent= new Intent(LoginActivity.this,MainActivity.class);
                    startActivityForResult(intent,1);
                }else{
                    Toast.makeText(LoginActivity.this,"用户不存在或者密码错误",Toast.LENGTH_SHORT).show();
                }
            }
        });

        pressRegister.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                Intent intent= new Intent(LoginActivity.this,RegisterActivity.class);
                startActivity(intent);
            }
        });
    }
    //密码可见性按钮监听
    private class ToggleButtonClick implements CompoundButton.OnCheckedChangeListener{

        @Override
        public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {

            //判断事件源的选中状态
            if (isChecked){
                //显示密码
                passwordEdit.setTransformationMethod(HideReturnsTransformationMethod.getInstance());
            }else {
                // 隐藏密码
                passwordEdit.setTransformationMethod(PasswordTransformationMethod.getInstance());

            }
            //每次显示或者关闭时，密码显示编辑的线不统一在最后，下面是为了统一
            passwordEdit.setSelection(passwordEdit.length());
        }
    }

}
